"""Cross-token DCT: interactions between a feature at earlier tokens and a feature at the current token.

A factor (l, r) pushes P l at every earlier position 1..t-1 and P r at position t of the residual stream entering block S
(after the block's λ mix), with P = Σ^½ the square root of the residual covariance Σ of a pool of windows, so every push is
one data standard deviation (unit-norm pushes collapse onto single neurons and heads). The cross term
C = ∂α∂β (readout at t) is scored as
    fisher: E = Σ_rows ½ Var_{p_t}(C)   on the next-token logits (the interaction's second-order KL, blocks S..17)
    euclid: E = Σ_rows ½ (u·C)²         on the residual stream entering block K, u an output direction
E is quadratic in each of u, l and r, so (u, l, r) <- normalize(∇E) is higher-order power iteration. Factors are fitted as
a census of independent random starts (--mode census: the end points are the attractors) or jointly with QR on L and R
(--mode joint, as in dct.py). Rows are (window, t) for t = 2..11 of 12-token windows sampled from the model itself
(--data model), WikiText or the model's bigram chain; Σ comes from a separate pool of windows of the same kind.

--score removes each feature's natural component on held-out WikiText windows (natural ablation) and reports the KL of
removing the interaction alone, KL(p || softmax(Z_a + Z_b - Z_ab)), against variance-matched controls (top principal
components with l = r, random directions drawn from Σ).

Usage: python xtok.py [--data model] [--S 12] [--kind fisher | --kind euclid --K 16] [--mode census] [--windows 16]
                      [--start 300] [--score]
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for path in (str(ROOT.parent), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np
import torch
from datasets import load_dataset
from scipy.sparse.csgraph import connected_components
from torch.func import grad_and_value, jvp, vmap
from torch.nn import functional as F
from transformers import GPT2Tokenizer

from tensor_model import apply_rotary_emb, load_tensor_gpt

device = torch.device("cuda")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model, _, _ = load_tensor_gpt("Elriggs/gpt2-swiglu-18l-9h-1152embd-v2", device)
model.requires_grad_(False)
blocks = model.transformer.h
n_layer, D = len(blocks), model.config.n_embd
H, hd = blocks[0].attn.n_head, blocks[0].attn.head_dim
length = 12
positions = torch.arange(length, device=device)
readouts = list(range(2, length))
out = ROOT / "cross_token"
with torch.no_grad():
    # fill the rotary caches outside any functorch transform (a cache created inside jvp escapes its level)
    for block in blocks:
        block.attn.rotary(torch.zeros(1, length, H, hd, device=device))
cos_cached, sin_cached = blocks[0].attn.rotary.cos_cached, blocks[0].attn.rotary.sin_cached
lm_weight = model.lm_head.weight
_windows = None


def wikitext_ids(start, count):
    global _windows
    if _windows is None:
        texts = [t.strip() for t in load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"] if len(t.strip()) > 40]
        _windows = [s[:length + 1] for s in (tokenizer(t).input_ids for t in texts) if len(s) >= length + 1]
    assert start + count <= len(_windows), f"only {len(_windows)} WikiText windows"
    return torch.tensor(_windows[start:start + count], device=device)


def bigram_ids(count, seed):
    next_ids, next_probs = torch.load(ROOT / "natural_dct" / "bigram" / "chain.pt")
    generator = torch.Generator(device=device).manual_seed(seed)
    token = torch.full((count,), tokenizer.eos_token_id, device=device)
    sequence = []
    for _ in range(length + 1):
        token = next_ids[token, torch.multinomial(next_probs[token].float(), 1, generator=generator)[:, 0]]
        sequence.append(token)
    return torch.stack(sequence, 1)


@torch.no_grad()
def model_ids(count, seed, batch=256):
    # temperature-1 samples from the model after <|endoftext|> (dropped)
    generator = torch.Generator(device=device).manual_seed(seed)
    samples = []
    for s in range(0, count, batch):
        sequence = torch.full((min(batch, count - s), 1), tokenizer.eos_token_id, device=device)
        for _ in range(length + 1):
            x = F.rms_norm(model.transformer.wte(sequence), (D,))
            x0, v1 = x, None
            for block in blocks:
                x, v1 = block(x, v1, x0)
            p = torch.softmax(logits(x[:, -1].float()), -1)
            sequence = torch.cat([sequence, torch.multinomial(p, 1, generator=generator)], 1)
        samples.append(sequence[:, 1:])
    return torch.cat(samples)


def sample_ids(data, start, count):
    # windows of length + 1 tokens; for bigram / model data, start offsets the sampling seed
    if data == "wikitext":
        return wikitext_ids(start, count)
    return bigram_ids(count, 2000 + start) if data == "bigram" else model_ids(count, 5000 + start)


def pool_ids(data, count):
    return wikitext_ids(700, count) if data == "wikitext" else bigram_ids(count, 9000) if data == "bigram" else model_ids(count, 9000)


@torch.no_grad()
def states(ids, S):
    s = model.state_before_block(ids[:, :length], S)
    x, initial, first = (s[k].float() for k in ("values", "initial_values", "first_values"))
    return blocks[S].lambdas[0] * x + blocks[S].lambdas[1] * initial, initial, first


def attention(attn, x, first):
    B, T, _ = x.shape
    q, k, v = (proj(x).view(B, T, H, hd) for proj in (attn.c_q, attn.c_k, attn.c_v))
    v = (1 - attn.lamb) * v + attn.lamb * first.view_as(v)
    cos, sin = cos_cached[None, :T, None, :], sin_cached[None, :T, None, :]
    q = apply_rotary_emb(F.rms_norm(q, (hd,)), cos, sin).transpose(1, 2)
    k = apply_rotary_emb(F.rms_norm(k, (hd,)), cos, sin).transpose(1, 2)
    causal = torch.ones(T, T, device=x.device, dtype=torch.bool).tril()
    pattern = (q @ k.transpose(-1, -2) / hd ** 0.5).masked_fill(~causal, float("-inf")).softmax(-1)
    return attn.c_proj((pattern @ v.transpose(1, 2)).transpose(1, 2).reshape(B, T, D))


def mlp(m, x):
    # explicit sigmoid: the fused silu breaks nested forward-mode AD
    g = m.left(x)
    return m.Down(g * torch.sigmoid(g) * m.right(x)) + m.Down_bias


def run(x, initial, first, S, K):
    # blocks S..K-1 from the mixed residual input of block S
    for b in range(S, K):
        block = blocks[b]
        if b > S:
            x = block.lambdas[0] * x + block.lambdas[1] * initial
        x = x + attention(block.attn, F.rms_norm(x, (D,)), first)
        x = x + mlp(block.mlp, F.rms_norm(x, (D,)))
    return x


def logits(h):
    return 30 * torch.tanh(model.lm_head(F.rms_norm(h, (D,))).float() / 30)


def masks(t):
    T = t + 1
    return (((positions[:T] >= 1) & (positions[:T] < t)).float()[None, :, None],
            (positions[:T] == t).float()[None, :, None])


def push_metric(m):
    # P = Σ^½ (scaled to unit mean diagonal) from the residual inputs at positions 1..11
    Z = m[:, 1:].reshape(-1, D).double()
    eigenvalues, eigenvectors = torch.linalg.eigh(torch.cov(Z.T))
    P = (eigenvectors * eigenvalues.clamp_min(0).sqrt()) @ eigenvectors.T
    return (P / P.diag().mean().sqrt()).float(), eigenvalues.float(), eigenvectors.float(), Z.mean(0).float()


class Rows:
    """Block-S residual inputs of the fitting windows and, for the fisher readout, each row's top-k next-token distribution."""

    def __init__(self, ids, S, kind, K, topk=512):
        self.ids, self.S, self.kind, self.K, self.topk = ids, S, kind, K, topk
        self.m, self.initial, self.first = states(ids, S)
        self.count = len(ids)
        self.aux = {}
        if kind == "fisher":
            with torch.no_grad():
                for s in range(0, self.count, 64):
                    Z = logits(run(self.m[s:s + 64], self.initial[s:s + 64], self.first[s:s + 64], S, n_layer))
                    for t in readouts:
                        p, index = torch.softmax(Z[:, t], -1).topk(topk, -1)
                        self.aux.setdefault(t, []).append(torch.cat([p, index.float()], -1))
            self.aux = {t: torch.cat(v) for t, v in self.aux.items()}


def make_step(kind, S, K, t, P, chunk, topk):
    earlier, here = masks(t)

    def step(u, l, r, m, initial, first, aux):
        if kind == "fisher":
            p, W = aux[:, :topk], lm_weight[aux[:, topk:].long()].float()
            head = lambda h: 30 * torch.tanh(torch.einsum("bd,bkd->bk", F.rms_norm(h[:, t], (D,)), W) / 30)
        else:
            head = lambda h: h[:, t]

        def objective(u, l, r):
            zero = torch.zeros_like(m)
            f = lambda z: head(run(z, initial, first, S, K))
            inner = lambda b: jvp(lambda a: f(m + a + b), (zero,), ((earlier * (P @ l)).expand_as(m),))[1]
            C = jvp(inner, (zero,), ((here * (P @ r)).expand_as(m),))[1]
            if kind == "fisher":
                return 0.5 * ((p * C.square()).sum(-1) - (p * C).sum(-1).square()).sum()
            return 0.5 * (C @ u).square().sum()

        (gu, gl, gr), value = grad_and_value(objective, argnums=(0, 1, 2))(u, l, r)
        return gu, gl, gr, value

    return vmap(step, in_dims=(1, 1, 1, None, None, None, None), out_dims=(1, 1, 1, 0), chunk_size=chunk)


class Fitter:
    def __init__(self, rows, P, factors=16, batch=32, chunk=8):
        self.rows, self.factors, self.batch = rows, factors, batch
        self.steps = {t: make_step(rows.kind, rows.S, rows.K, t, P, chunk, rows.topk) for t in readouts}

    def sweep(self, U, L, R):
        rows = self.rows
        totals = [torch.zeros_like(U), torch.zeros_like(L), torch.zeros_like(R), torch.zeros(self.factors, device=device)]
        for t in readouts:
            T = t + 1
            for s in range(0, rows.count, self.batch):
                part = slice(s, s + self.batch)
                aux = rows.aux[t][part] if rows.aux else torch.zeros(1, device=device)
                outputs = self.steps[t](U, L, R, rows.m[part, :T], rows.initial[part, :T], rows.first[part, :T], aux)
                for total, value in zip(totals, outputs):
                    total += value.detach()
        return totals

    def fit(self, seed, mode="census", iters=25, tolerance=1e-5, patience=4, log=print):
        generator = torch.Generator(device=device).manual_seed(seed)
        U, L, R = (F.normalize(torch.randn(D, self.factors, device=device, generator=generator), dim=0) for _ in range(3))
        orthogonalize = (lambda Z: torch.linalg.qr(Z).Q) if mode == "joint" else (lambda Z: Z)
        trace, flat, begin = [], 0, time.time()
        rows = self.rows.count * len(readouts)
        for it in range(iters):
            L, R = orthogonalize(L), orthogonalize(R)
            gU, gL, gR, E = self.sweep(U, L, R)
            if self.rows.kind == "euclid":
                U = F.normalize(gU, dim=0)
            new_L, new_R = F.normalize(gL, dim=0), F.normalize(gR, dim=0)
            drift = torch.minimum((new_L * L).sum(0).abs(), (new_R * R).sum(0).abs())
            L, R = new_L, new_R
            trace.append(E.cpu().numpy())
            total = trace[-1].sum()
            flat = flat + 1 if len(trace) > 1 and abs(total - trace[-2].sum()) <= tolerance * abs(total) else 0
            if log and (it % 5 == 4 or it == 0 or flat >= patience):
                log(f"  sweep {it + 1:2d}: objective per row (mean over factors) {total / rows / self.factors:.4e}, min |cos| to "
                    f"previous {drift.min().item():.5f}, {time.time() - begin:.0f} s")
            if flat >= patience:
                break
        L, R = orthogonalize(L), orthogonalize(R)
        E = self.sweep(U, L, R)[3]
        order = torch.argsort(E, descending=True)
        return {"U": U[:, order].cpu(), "L": L[:, order].cpu(), "R": R[:, order].cpu(), "E": E[order].cpu(),
                "drift": drift[order].cpu(), "trace": np.array(trace), "iterations": len(trace), "seconds": time.time() - begin}


def context(ids, w, p, before=5):
    text = tokenizer.decode(ids[w, max(1, p - before):p]) + "[" + tokenizer.decode(ids[w, p:p + 1]) + "]" + tokenizer.decode(ids[w, p + 1:p + 2])
    return text.replace("\n", "⏎")


def fires_on(ids, deviation, v, k=5):
    # contexts with the largest activity along v (one per distinct current token), both signs
    a = deviation @ v
    sd = a.std()
    lines = []
    for sign, mark in ((1, "+"), (-1, "−")):
        shown, seen = [], set()
        for n in (sign * a).flatten().topk(min(400, a.numel())).indices.tolist():
            w, p = divmod(n, length - 1)
            p += 1
            if int(ids[w, p]) in seen:
                continue
            seen.add(int(ids[w, p]))
            shown.append(f"{context(ids, w, p, 4)} ({a[w, p - 1] / sd:+.1f}σ)")
            if len(shown) == k:
                break
        lines.append(f"{mark} " + " | ".join(shown))
    return lines


class Scorer:
    """Natural ablations on held-out WikiText windows (0..count-1); rows ordered (t, window)."""

    def __init__(self, S, count=256):
        self.S, self.count = S, count
        self.ids = wikitext_ids(0, count)
        self.m, self.initial, self.first = states(self.ids, S)
        self.mu = self.m[:, 1:].reshape(-1, D).mean(0)
        with torch.no_grad():
            self.Z = torch.cat([logits(run(self.m[s:s + 64], self.initial[s:s + 64], self.first[s:s + 64], S, n_layer))
                                for s in range(0, count, 64)])

    @torch.no_grad()
    def __call__(self, L, R):
        # KL of removing the interaction, of removing l (r present) and of removing l with r already removed
        keys = ("interaction", "l", "l without r")
        result = {k: [[] for _ in L] for k in keys}
        for t in readouts:
            T = t + 1
            earlier, here = masks(t)
            for s in range(0, self.count, 128):
                m, initial, first = self.m[s:s + 128, :T], self.initial[s:s + 128, :T], self.first[s:s + 128, :T]
                lp = torch.log_softmax(self.Z[s:s + 128, t], -1)
                kl = lambda ref, Z: (ref.exp() * (ref - torch.log_softmax(Z, -1))).sum(-1)
                deviation = m - self.mu
                for j in range(len(L)):
                    a = earlier * (deviation @ L[j])[..., None] * L[j]
                    b = here * (deviation @ R[j])[..., None] * R[j]
                    Za, Zb, Zab = (logits(run(z, initial, first, self.S, n_layer)[:, t]) for z in (m - a, m - b, m - a - b))
                    result["interaction"][j].append(kl(lp, Za + Zb - Zab))
                    result["l"][j].append(kl(lp, Za))
                    result["l without r"][j].append(kl(torch.log_softmax(Zb, -1), Zab))
        return {k: torch.stack([torch.cat(v) for v in values]).cpu().numpy() for k, values in result.items()}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", choices=["model", "wikitext", "bigram"], default="model")
    parser.add_argument("--S", type=int, default=12, help="block whose residual input is pushed")
    parser.add_argument("--kind", choices=["fisher", "euclid"], default="fisher")
    parser.add_argument("--K", type=int, default=16, help="readout block for --kind euclid")
    parser.add_argument("--mode", choices=["census", "joint"], default="census")
    parser.add_argument("--factors", type=int, default=16, help="random starts (census) or factors (joint)")
    parser.add_argument("--windows", type=int, default=16, help="fitting windows of 12 tokens")
    parser.add_argument("--start", type=int, default=300, help="first WikiText window, or sampling-seed offset for model/bigram")
    parser.add_argument("--pool", type=int, default=1000, help="windows for the covariance (WikiText pool: windows 700..)")
    parser.add_argument("--iters", type=int, default=25)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--chunk", type=int, default=8, help="factors per vmap chunk (lower it for small S)")
    parser.add_argument("--topk", type=int, default=512, help="tokens per row in the fisher readout")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report", type=int, default=4, help="attractors / factors to describe")
    parser.add_argument("--score", action="store_true", help="held-out WikiText natural-ablation scores and controls")
    args = parser.parse_args()
    assert args.kind == "fisher" or args.K > args.S, "--K must be above --S"
    if args.data == "wikitext":
        assert args.start >= 256 and (args.start + args.windows <= 700 or args.start >= 700 + args.pool), \
            "fitting windows must avoid the evaluation windows (0-255) and the covariance pool (700..)"
    return args


def main():
    args = parse_args()
    torch.set_float32_matmul_precision("high")
    out.mkdir(exist_ok=True)
    K = args.K if args.kind == "euclid" else n_layer
    readout = "fisher" if args.kind == "fisher" else f"u{args.K}"
    name = f"S{args.S}_{readout}_{args.data}_{args.mode}_w{args.windows}_st{args.start}_seed{args.seed}"
    begin = time.time()
    pool = pool_ids(args.data, args.pool)
    pool_m = states(pool, args.S)[0]
    P, eigenvalues, eigenvectors, mu = push_metric(pool_m)
    rows = Rows(sample_ids(args.data, args.start, args.windows), args.S, args.kind, K, args.topk)
    print(f"{name}: {args.windows} fitting windows ({args.windows * len(readouts)} rows), covariance from {args.pool} {args.data} "
          f"windows; setup {time.time() - begin:.0f} s", flush=True)
    fit = Fitter(rows, P, args.factors, args.batch, args.chunk).fit(args.seed, args.mode, args.iters, args.tol)
    L, R = (F.normalize(fit[k].T.to(device) @ P, dim=1) for k in "LR")
    if args.mode == "census":
        similarity = ((L @ L.T).abs() * (R @ R.T).abs()).cpu().numpy()
        label = connected_components(similarity > 0.8, directed=False)[1]
    else:
        label = np.arange(args.factors)
    sizes = np.bincount(label)
    groups = sorted(range(len(sizes)), key=lambda g: (-sizes[g], np.nonzero(label == g)[0][0]))
    representatives = [int(np.nonzero(label == g)[0][0]) for g in groups]
    per_row = fit["E"].numpy() / (args.windows * len(readouts))
    print(f"\n{fit['iterations']} sweeps, {fit['seconds']:.0f} s" + (f"; {len(groups)} attractors from {args.factors} starts, basin "
          f"sizes {[int(sizes[g]) for g in groups]}" if args.mode == "census" else ""))
    deviation = (pool_m - mu)[:, 1:]
    for rank, j in enumerate(representatives[:args.report]):
        label_text = f"attractor {rank} ({sizes[groups[rank]]} starts)" if args.mode == "census" else f"factor {j}"
        print(f"\n{label_text}: objective per row {per_row[j]:.3e}; |cos l·r| {(L[j] @ R[j]).abs().item():.2f}")
        for side, v in (("l (earlier tokens)", L[j]), ("r (current token)", R[j])):
            lines = fires_on(pool, deviation, v)
            print(f"  {side:18s} {lines[0]}\n  {'':18s} {lines[1]}")
    result = {**fit, "P": P.cpu(), "label": label, "representatives": representatives, "args": vars(args)}
    if args.score:
        torch.set_float32_matmul_precision("highest")
        scorer = Scorer(args.S)
        chosen = representatives[:args.report]
        scores = scorer(L[chosen], R[chosen])
        generator = torch.Generator(device=device).manual_seed(11)
        random_pushes = lambda: F.normalize(torch.randn(16, D, device=device, generator=generator) @ P, dim=1)
        top_pcs = eigenvectors.flip(1).T[:16]
        controls = [scorer(top_pcs, top_pcs)["interaction"].mean(1).max(), scorer(random_pushes(), random_pushes())["interaction"].mean(1).max()]
        control = float(max(controls))
        print(f"\nheld-out WikiText ({scorer.count} windows, {scorer.count * len(readouts)} rows); best variance-matched control "
              f"mean KL_int {control:.2e}")
        for rank, j in enumerate(chosen):
            ki, kl_l, kl_l_alone = (scores[k][rank] for k in ("interaction", "l", "l without r"))
            active = ki > 0.01
            and_ness = (f"; on active rows: interaction / l's effect {ki[active].mean() / kl_l[active].mean():.2f}, l's effect without r / "
                        f"with r {kl_l_alone[active].mean() / kl_l[active].mean():.2f}") if active.any() else ""
            print(f"  {'attractor' if args.mode == 'census' else 'factor'} {rank if args.mode == 'census' else j}: mean KL_int "
                  f"{ki.mean():.2e} ({ki.mean() / control:.1f}x control), active (KL_int > 0.01) on {active.mean():.1%} of rows{and_ness}")
            for i in np.argsort(-ki)[:3]:
                ti, w = divmod(int(i), scorer.count)
                print(f"      KL_int {ki[i]:.3f}  {context(scorer.ids, w, readouts[ti], 8)}")
        result["scores"], result["control"] = scores, control
    torch.save(result, out / f"{name}.pt")
    print(f"\nsaved {out / name}.pt ({time.time() - begin:.0f} s)")


if __name__ == "__main__":
    main()
