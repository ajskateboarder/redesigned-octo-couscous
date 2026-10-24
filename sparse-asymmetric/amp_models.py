"""Architecture adapters for downstream_amp.py.

Each adapter exposes
    dev, D, n_layer, tokenizer
    bigram_chain()                   -> path of the model's bigram chain (top-64 continuations per token), built if missing
    capture(ids, L)                  -> x, r, ctx: the normalised input of MLP L, the residual it is added to, and any
                                        extra state the later blocks need (a tuple of tensors with a leading batch dim)
    downstream(x, r, ctx, theta, L)  -> final residual, with the input of MLP L shifted by theta
    logits(h)                        -> next-token logits from final residuals
    head(h, idx)                     -> logits of the tokens idx (b, p, k) from final residuals h (b, p, D)
    next_logits(seq)                 -> next-token logits after the token sequences seq (b, t)
Optional (downstream_amp.py's defaults in brackets): max_layer, the last MLP that can be fitted [n_layer - 2]; bos, the
token sampled windows start from [50256]; keep_bos, whether windows keep it at position 0 [False]; special, token ids that
disqualify a sampled window [{50256}]; retokenize(span), turning a GPT-2 token span into a window [absent: GPT-2 ids are
used as they are].
    mlp_parts(L)                     -> MLP L as hidden(x) (the neuron activations), W_out (H, D), bias, after(h, ctx)
                                        (the later blocks, from the residual after MLP L), kind ("gelu", "relu2", "sq",
                                        "bilinear" or "gated") and for threshold MLPs pre(x), w_in (D, H), for bilinear
                                        ones left(x), right(x), W_left, W_right (H, D)

swiglu: Elriggs/gpt2-swiglu-18l-9h-1152embd-v2 (via cross_token/xtok.py; RMSNorm, value embeddings, λ residual mix).
gpt2:   GPT-2 small (12 layers, D 768, GELU MLP, pre-LayerNorm). The MLP input is the output of ln_2, affine included.
gpt2_noln: apollo-research/gpt2_noLN, GPT-2 small fine-tuned with LayerNorm removed. Its LNs are stored as eps 1e12 with
        scale 1e6, i.e. x ↦ (x - mean x)·w/1e6 + b; they are applied in exactly that linear form, so the MLP input is the
        mean-centred residual and the logits are linear in the final residual.
gelu_1l: NeelNanda/GELU_1L512W_C4_Code, a one-layer GELU toy model (D 512, 8 heads, 2048 neurons, pre-LayerNorm, learned
        positions, no unembedding bias; trained with AdamW on C4 + code, with <|BOS|> at the start of every sequence). It
        has its own tokenizer (NeelNanda/gpt-neox-tokenizer-digits), so OpenWebText windows are retokenised from the text
        of a 2·WIN-token GPT-2 span and get <|BOS|> at position 0. The MLP input is ln2's output, affine included; after the
        MLP only ln_final and the unembedding remain. gelu_2l: NeelNanda/GELU_2L512W_C4_Code, the same with two layers
        (layer 0's MLP is followed by block 1, layer 1's only by ln_final and the unembedding).
gelu_1l+<name>: gelu_1l with its MLP replaced by downstream_amp/gate_bound/mlp/<name>.pt, an MLP of another width /
        activation retrained by amp_mlp_width.py with everything else frozen (d_mlp scaling at fixed d_model).
relu2_12l, sq_12l, bilinear_12l, swiglu_12l: Elriggs' 12-layer models (D 768, 6 heads, tensor_model.py), one training recipe
        (modded-nanogpt, Muon) that differs only in the MLP: ReLU², x², bilinear, SwiGLU. Same block structure as swiglu
        (weightless RMSNorm, value mixing with the first block's values, λ residual mix); the MLP input is the RMSNorm output.
"""
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn import functional as F

ROOT_DIR = Path(__file__).resolve().parent
BIGRAM = ROOT_DIR / "dct_etc" / "bigram"
RETRAINED = ROOT_DIR / "downstream_amp" / "gate_bound" / "mlp"     # gelu_1l MLPs retrained by amp_mlp_width.py


def swiglu(win):
    _argv, sys.argv = sys.argv, [sys.argv[0]]
    import xtok as X
    sys.argv = _argv
    for b in X.blocks:
        b.attn.rotary(torch.zeros(1, win, X.H, X.hd, device=X.device))
    X.cos_cached, X.sin_cached = X.blocks[0].attn.rotary.cos_cached, X.blocks[0].attn.rotary.sin_cached
    D = X.D

    @torch.no_grad()
    def capture(ids, L):
        BL = X.blocks[L]
        st = X.model.state_before_block(ids, L)
        x, ini, fir = (st[k].float() for k in ("values", "initial_values", "first_values"))
        m = BL.lambdas[0] * x + BL.lambdas[1] * ini
        r = m + X.attention(BL.attn, F.rms_norm(m, (D,)), fir)
        return F.rms_norm(r, (D,)), r, (ini, fir)

    def downstream(x, r, ctx, theta, L):
        ini, fir = ctx
        BL, BN = X.blocks[L], X.blocks[L + 1]
        h = r + X.mlp(BL.mlp, x + theta)
        return X.run(BN.lambdas[0] * h + BN.lambdas[1] * ini, ini, fir, L + 1, X.n_layer)

    def head(h, idx):
        return 30 * torch.tanh(torch.einsum("bpd,bpkd->bpk", F.rms_norm(h, (D,)), X.lm_weight[idx].float()) / 30)

    def next_logits(seq):
        x = F.rms_norm(X.model.transformer.wte(seq), (D,))
        x0, v1 = x, None
        for blk in X.blocks:
            x, v1 = blk(x, v1, x0)
        return X.logits(x[:, -1].float())

    def mlp_parts(L):
        m, BN = X.blocks[L].mlp, X.blocks[L + 1]

        def hidden(x):
            u = m.left(x)
            return u * torch.sigmoid(u) * m.right(x)

        after = lambda h, ctx: X.run(BN.lambdas[0] * h + BN.lambdas[1] * ctx[0], ctx[0], ctx[1], L + 1, X.n_layer)
        return dict(hidden=hidden, W_out=m.Down.weight.T.float(), bias=m.Down_bias, after=after, kind="gated",
                    left=m.left, right=m.right, W_left=m.Left.weight, W_right=m.Right.weight)

    return SimpleNamespace(dev=X.device, D=D, n_layer=X.n_layer, tokenizer=X.tokenizer,
                           bigram_chain=lambda: BIGRAM / "chain.pt", mlp_parts=mlp_parts,
                           capture=capture, downstream=downstream, logits=X.logits, head=head, next_logits=next_logits)


def gpt2(win, repo="gpt2", noln=False):
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    dev = torch.device("cuda")
    model = GPT2LMHeadModel.from_pretrained(repo).to(dev).eval().requires_grad_(False)
    tr, blocks = model.transformer, model.transformer.h
    D, H = model.config.n_embd, model.config.n_head
    hd = D // H
    causal = torch.ones(win, win, device=dev, dtype=torch.bool).tril()
    if noln:
        ln = lambda m, x: (x - x.mean(-1, keepdim=True)) * (m.weight / m.eps ** 0.5) + m.bias
    else:
        ln = lambda m, x: F.layer_norm(x, (D,), m.weight, m.bias, m.eps)

    # written out (Conv1D weights are (in, out)) so that nested vmap / grad / jvp go through
    def attention(a, x):
        B, T, _ = x.shape
        q, k, v = ((x @ a.c_attn.weight + a.c_attn.bias).split(D, -1))
        q, k, v = (t.reshape(B, T, H, hd).transpose(1, 2) for t in (q, k, v))
        pattern = (q @ k.transpose(-1, -2) / hd ** 0.5).masked_fill(~causal[:T, :T], float("-inf")).softmax(-1)
        return (pattern @ v).transpose(1, 2).reshape(B, T, D) @ a.c_proj.weight + a.c_proj.bias

    gelu = lambda u: 0.5 * u * (1 + torch.tanh(math.sqrt(2 / math.pi) * (u + 0.044715 * u ** 3)))   # gelu_new, as trained

    def mlp(m, x):
        return gelu(x @ m.c_fc.weight + m.c_fc.bias) @ m.c_proj.weight + m.c_proj.bias

    def block(b, h):
        h = h + attention(b.attn, ln(b.ln_1, h))
        return h + mlp(b.mlp, ln(b.ln_2, h))

    def embed(ids):
        return tr.wte(ids) + tr.wpe(torch.arange(ids.shape[1], device=dev))

    @torch.no_grad()
    def capture(ids, L):
        h = embed(ids)
        for b in blocks[:L]:
            h = block(b, h)
        r = h + attention(blocks[L].attn, ln(blocks[L].ln_1, h))
        return ln(blocks[L].ln_2, r), r, ()

    def downstream(x, r, ctx, theta, L):
        h = r + mlp(blocks[L].mlp, x + theta)
        for b in blocks[L + 1:]:
            h = block(b, h)
        return h

    def logits(h):
        return ln(tr.ln_f, h) @ tr.wte.weight.T

    def head(h, idx):
        return torch.einsum("bpd,bpkd->bpk", ln(tr.ln_f, h), tr.wte.weight[idx])

    def next_logits(seq):
        h = embed(seq)
        for b in blocks:
            h = block(b, h)
        return logits(h[:, -1])

    def mlp_parts(L):
        m = blocks[L].mlp
        pre = lambda x: x @ m.c_fc.weight + m.c_fc.bias

        def after(h, ctx):
            for b in blocks[L + 1:]:
                h = block(b, h)
            return h

        return dict(hidden=lambda x: gelu(pre(x)), W_out=m.c_proj.weight.float(), bias=m.c_proj.bias, after=after,
                    kind="gelu", pre=pre, w_in=m.c_fc.weight)

    bigram_chain = lambda: _bigram_chain(f"chain_{'gpt2_noln' if noln else 'gpt2'}.pt", next_logits, model.config.vocab_size, dev)
    return SimpleNamespace(dev=dev, D=D, n_layer=len(blocks), tokenizer=GPT2Tokenizer.from_pretrained("gpt2"),
                           bigram_chain=bigram_chain, capture=capture, downstream=downstream, logits=logits, head=head,
                           next_logits=next_logits, mlp_parts=mlp_parts, hf=model, block=block, embed=embed)


def _bigram_chain(name, next_logits, vocab, dev):
    chain = BIGRAM / name
    if not chain.exists():
        # as dct_etc/bigram_dct.py: next-token distribution from each single token, truncated to its top-64 continuations
        ids_, probs_ = [], []
        with torch.no_grad():
            for batch in torch.arange(vocab, device=dev).split(2048):
                top = next_logits(batch[:, None])[:, :vocab].softmax(-1).topk(64, dim=-1)
                ids_.append(top.indices)
                probs_.append(top.values / top.values.sum(-1, keepdim=True))
        chain.parent.mkdir(parents=True, exist_ok=True)
        torch.save((torch.cat(ids_), torch.cat(probs_)), chain)
    return chain


TENSOR = {"relu2_12l": "Elriggs/gpt2-12l-6h-768embd", "sq_12l": "Elriggs/gpt2-mlp2-12l-6h-768embd",
          "bilinear_12l": "Elriggs/gpt2-bilinear-12l-6h-768embd",
          "swiglu_12l": "Elriggs/gpt2-bilinear-swiglu-12l-6h-768embd"}


def tensor(win, name):
    from transformers import GPT2Tokenizer
    from tensor_model import apply_rotary_emb, load_tensor_gpt
    dev = torch.device("cuda")
    model, cfg, _ = load_tensor_gpt(TENSOR[name], dev)
    model.requires_grad_(False)
    assert not cfg.squared_attn and not cfg.bilinear_attn
    blocks = model.transformer.h
    D, H = cfg.n_embd, cfg.n_head
    hd = D // H
    with torch.no_grad():
        # fill the rotary caches outside any functorch transform (as in cross_token/xtok.py); kept in the model's bf16
        for b in blocks:
            b.attn.rotary(torch.zeros(1, win, H, hd, device=dev))
    cos_c, sin_c = blocks[0].attn.rotary.cos_cached, blocks[0].attn.rotary.sin_cached
    causal = torch.ones(win, win, device=dev, dtype=torch.bool).tril()
    norm = lambda x: F.rms_norm(x, (D,))

    # written out (no fused attention or silu) so that nested vmap / grad / jvp go through
    def attention(a, x, first):
        B, T, _ = x.shape
        q, k, v = (proj(x).view(B, T, H, hd) for proj in (a.c_q, a.c_k, a.c_v))
        v = (1 - a.lamb) * v + a.lamb * first.view_as(v)
        cos, sin = cos_c[None, :T, None, :], sin_c[None, :T, None, :]
        q = apply_rotary_emb(F.rms_norm(q, (hd,)), cos, sin).transpose(1, 2)
        k = apply_rotary_emb(F.rms_norm(k, (hd,)), cos, sin).transpose(1, 2)
        pattern = (q @ k.transpose(-1, -2) / hd ** 0.5).masked_fill(~causal[:T, :T], float("-inf")).softmax(-1)
        return a.c_proj((pattern @ v.transpose(1, 2)).transpose(1, 2).reshape(B, T, D))

    def hidden_of(m, x):
        if cfg.bilinear:
            u = m.left(x)
            return (u * torch.sigmoid(u) if cfg.gated else u) * m.right(x)
        u = m.c_fc(x)
        return u.square() if cfg.squared_mlp else F.relu(u).square()

    def mlp(m, x):
        h = hidden_of(m, x)
        return m.Down(h) + m.Down_bias if cfg.bilinear else m.c_proj(h) + m.bias

    def block(b, h, x0, first):
        h = b.lambdas[0] * h + b.lambdas[1] * x0
        h = h + attention(b.attn, norm(h), first)
        return h + mlp(b.mlp, norm(h))

    @torch.no_grad()
    def capture(ids, L):
        assert L >= 1, "block 0's attention defines the first-block values the later blocks mix in"
        st = model.state_before_block(ids, L)
        x, x0, first = (st[k].float() for k in ("values", "initial_values", "first_values"))
        BL = blocks[L]
        m_ = BL.lambdas[0] * x + BL.lambdas[1] * x0
        r = m_ + attention(BL.attn, norm(m_), first)
        return norm(r), r, (x0, first)

    def run_after(h, ctx, L):
        for b in blocks[L + 1:]:
            h = block(b, h, *ctx)
        return h

    def downstream(x, r, ctx, theta, L):
        return run_after(r + mlp(blocks[L].mlp, x + theta), ctx, L)

    def logits(h):
        return 30 * torch.tanh(model.lm_head(norm(h)).float() / 30)

    def head(h, idx):
        return 30 * torch.tanh(torch.einsum("bpd,bpkd->bpk", norm(h), model.lm_head.weight[idx].float()) / 30)

    def next_logits(seq):
        return model.logits(seq)[:, -1]

    def mlp_parts(L):
        m = blocks[L].mlp
        parts = dict(hidden=lambda x: hidden_of(m, x), after=lambda h, ctx: run_after(h, ctx, L))
        if cfg.bilinear:
            parts.update(W_out=m.Down.weight.T.float(), bias=m.Down_bias, kind="gated" if cfg.gated else "bilinear",
                         left=m.left, right=m.right, W_left=m.Left.weight, W_right=m.Right.weight)
        else:
            parts.update(W_out=m.c_proj.weight.T.float(), bias=m.bias, kind="sq" if cfg.squared_mlp else "relu2",
                         pre=m.c_fc, w_in=m.c_fc.weight.T)
        return parts

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    return SimpleNamespace(dev=dev, D=D, n_layer=len(blocks), tokenizer=tokenizer,
                           bigram_chain=lambda: _bigram_chain(f"chain_{name}.pt", next_logits, len(tokenizer), dev),
                           capture=capture, downstream=downstream, logits=logits, head=head, next_logits=next_logits,
                           mlp_parts=mlp_parts, model=model)


NEEL = {"gelu_1l": "NeelNanda/GELU_1L512W_C4_Code", "gelu_2l": "NeelNanda/GELU_2L512W_C4_Code"}


def neel_gelu(win, name, retrained=None):
    from transformers import AutoTokenizer, GPT2Tokenizer
    from huggingface_hub import hf_hub_download
    dev = torch.device("cuda")
    sd = {k: v.to(dev) for k, v in torch.load(hf_hub_download(NEEL[name], "model_final.pth"), map_location="cpu",
                                               weights_only=True).items()}
    act = F.gelu
    if retrained is not None:
        # gelu_1l+<checkpoint>: layer 0's MLP replaced by one retrained with everything else frozen (amp_mlp_width.py)
        ck = torch.load(RETRAINED / f"{retrained}.pt", map_location=dev, weights_only=False)
        sd.update({f"blocks.0.mlp.{k}": ck[k].float() for k in ("W_in", "b_in", "W_out", "b_out")})
        act = ACTS[ck["act"]]
    tok = AutoTokenizer.from_pretrained("NeelNanda/gpt-neox-tokenizer-digits")
    gpt2_tok = GPT2Tokenizer.from_pretrained("gpt2")
    D, H = sd["embed.W_E"].shape[1], sd["blocks.0.attn.W_Q"].shape[0]
    n_layer = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("blocks."))
    hd = D // H
    causal = torch.ones(win, win, device=dev, dtype=torch.bool).tril()
    ln = lambda x, name: F.layer_norm(x, (D,), sd[f"{name}.w"], sd[f"{name}.b"], 1e-5)

    def attention(l, x):
        a_ = lambda k: sd[f"blocks.{l}.attn.{k}"]
        T = x.shape[1]
        q, k, v = (torch.einsum("btd,ndh->bnth", x, a_(f"W_{c}")) + a_(f"b_{c}")[None, :, None] for c in "QKV")
        pattern = (q @ k.transpose(-1, -2) / hd ** 0.5).masked_fill(~causal[:T, :T], float("-inf")).softmax(-1)
        return torch.einsum("bnth,nhd->btd", pattern @ v, a_("W_O")) + a_("b_O")

    pre = lambda l: lambda x: x @ sd[f"blocks.{l}.mlp.W_in"] + sd[f"blocks.{l}.mlp.b_in"]
    hidden = lambda l: lambda x: act(pre(l)(x))                     # exact GELU, as TransformerLens' "gelu"

    def mlp(l, x):
        return hidden(l)(x) @ sd[f"blocks.{l}.mlp.W_out"] + sd[f"blocks.{l}.mlp.b_out"]

    def block(l, h):
        h = h + attention(l, ln(h, f"blocks.{l}.ln1"))
        return h + mlp(l, ln(h, f"blocks.{l}.ln2"))

    def after(h, L):
        for l in range(L + 1, n_layer):
            h = block(l, h)
        return h

    @torch.no_grad()
    def capture(ids, L):
        h = sd["embed.W_E"][ids] + sd["pos_embed.W_pos"][:ids.shape[1]]
        for l in range(L):
            h = block(l, h)
        r = h + attention(L, ln(h, f"blocks.{L}.ln1"))
        return ln(r, f"blocks.{L}.ln2"), r, ()

    def downstream(x, r, ctx, theta, L):
        return after(r + mlp(L, x + theta), L)

    def logits(h):
        return ln(h, "ln_final") @ sd["unembed.W_U"]

    def head(h, idx):
        return torch.einsum("bpd,bpkd->bpk", ln(h, "ln_final"), sd["unembed.W_U"].T[idx])

    def next_logits(seq):
        x, r, _ = capture(seq, 0)
        return logits(downstream(x, r, (), 0.0, 0))[:, -1]

    def retokenize(span):
        ids = tok(gpt2_tok.decode(span), add_special_tokens=False).input_ids
        return [tok.bos_token_id] + ids[:win - 1] if len(ids) >= win - 1 else None

    def mlp_parts(L):
        return dict(hidden=hidden(L), W_out=sd[f"blocks.{L}.mlp.W_out"].float(), bias=sd[f"blocks.{L}.mlp.b_out"],
                    after=lambda h, ctx: after(h, L), kind="gelu" if retrained is None else ck["act"], pre=pre(L), w_in=sd[f"blocks.{L}.mlp.W_in"])

    return SimpleNamespace(dev=dev, D=D, n_layer=n_layer, max_layer=n_layer - 1, tokenizer=tok, bos=tok.bos_token_id,
                           keep_bos=True, special={tok.bos_token_id, tok.eos_token_id, tok.pad_token_id},
                           retokenize=retokenize,
                           bigram_chain=lambda: _bigram_chain(f"chain_{name}.pt", next_logits, len(tok), dev),
                           capture=capture, downstream=downstream, logits=logits, head=head, next_logits=next_logits,
                           mlp_parts=mlp_parts, sd=sd)


def load(name, win):
    if name in NEEL:
        return neel_gelu(win, name)
    if name.startswith("gelu_1l+"):
        return neel_gelu(win, "gelu_1l", name.split("+", 1)[1])
    if name == "gpt2_noln":
        return gpt2(win, "apollo-research/gpt2_noLN", noln=True)
    if name in TENSOR:
        return tensor(win, name)
    return {"swiglu": swiglu, "gpt2": gpt2}[name](win)


ACTS = {"gelu": F.gelu, "relu": F.relu, "relu2": lambda u: F.relu(u).square()}
MODELS = ["swiglu", "gpt2", "gpt2_noln", *TENSOR, *NEEL, *(f"gelu_1l+{p.stem}" for p in sorted(RETRAINED.glob("*.pt")))]
