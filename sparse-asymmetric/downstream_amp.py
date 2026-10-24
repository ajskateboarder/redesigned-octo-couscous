"""
Rows ρ are (window, position) pairs, positions FIRST..WIN-1 of OpenWebText windows. z_ρ is the input to MLP `--layer`
(after the block's RMSNorm); μ, Σ its mean and covariance on the fitting rows. o and s are unit vectors in whitened
coordinates.
    channel gain  g_ρ(s) = ½ Var_{p_ρ}(J_ρ Σ^½ s) = sᵀ M_ρ s,   M_ρ = ½ Σ^½ J_ρᵀ (diag p_ρ - p_ρ p_ρᵀ) J_ρ Σ^½
                  J_ρ: Jacobian of the row's next-token logits w.r.t. a push into the MLP input at every position
                  (through the MLP and all later blocks); p_ρ: the model's next-token distribution (top-k tokens)
    normaliser    τ_ρ = mean gain of 8 random unit pushes (removes rows where every direction has a large effect)
    opener state  c_ρ(o) = (z_ρ - μ)ᵀ Σ^-½ o;  o "on" = top `--tail` of rows
    objective     A(o, s) = E[g(s)/τ | o on] - E[g(s)/τ | o off] = sᵀ M(o) s,   s ⟂ o
Fit: alternate a shifted power step in s (the top eigenvector of M(o), projected ⟂ o) with Adam steps on a soft version of
the on/off contrast in o, o restricted to the top `--ko` principal directions of z (0: unrestricted). Attractors are end
points reached by ≥ 2 of `--starts` random starts (|cos o|, |cos s| > 0.8).

Checks on held-out windows, per attractor:
    contrast        vs random (o, s ⟂ o) pairs
    causal          push off rows along Σ^½ o to the on-mean coordinate: share of the contrast reproduced (vs an equal random
                    push that leaves o's coordinate unchanged); and the gain ratio of s vs of random s
    entropy null    contrast within quintiles of next-token entropy / overall
    content null    contrast within strata of "next token is a content word" / overall
Description: top / bottom contexts of o and s (one per document), and the tokens the opened channel promotes and
suppresses (mean change of next-token log-probs from pushing s, on rows minus off rows).

--model gpt2 runs the same objective on GPT-2 small (GELU MLPs, pre-LayerNorm): z is then the output of the block's ln_2
(affine included), the actual input of c_fc. --model gpt2_noln uses apollo-research/gpt2_noLN (GPT-2 small fine-tuned
without LayerNorm): z is the mean-centred residual, linear in the residual stream. --model relu2_12l / sq_12l /
bilinear_12l / swiglu_12l use Elriggs' 12-layer models, trained with one recipe (Muon) that differs only in the MLP:
ReLU², x², bilinear, SwiGLU. --model gelu_1l --layer 0 uses a one-layer GELU toy model (NeelNanda/GELU_1L512W_C4_Code)
with its own tokenizer, --model gelu_2l --layer 0|1 its two-layer sibling; see amp_models.py.

Usage: python downstream_amp.py [--model swiglu|gpt2|gpt2_noln|relu2_12l|sq_12l|bilinear_12l|swiglu_12l|gelu_1l|gelu_2l] [--data owt|model|bigram] [--layer 8] [--windows 480] [--ko 16] [--starts 32] [--iters 40] [--draw 0]
Output: downstream_amp/[{model}_]L{layer}_K{ko}_{data}_d{draw}_n{windows}[_t{tail}].pt

Further checks on a saved run (each rebuilds this script's rows from the run's saved arguments; see amp_common.py):
    gate_opening.py      does one attractor's opener open another's channel (with the overlap / direct split)
    amp_neurons.py       per-neuron split of each contrast, and whether the gating is inside this MLP at all
    amp_neuron_openers.py  whether a gate's fitted opener is the direction its top neurons actually read
    amp_gate_anatomy.py  whether a gate's "open" rows are a single direction, a self-gate on s, or o AND s
    amp_opener_shape.py  where the openers sit in the PC spectrum and how clustered their coordinates are
    amp_pc_sparsity.py   whether sparse (transcoder) features lean on low-variance PCs (GPT-2 only)
    amp_transcoder.py    transcoder features as openers, and how attractor on-sets relate to them (GPT-2 only)
    amp_basis_alignment.py  (no run needed) how basis-aligned each model's MLP input is, across activations and optimizers
    amp_gate_bound.py    the dense-gate bound (gate_bound.md) on MLPs followed only by the readout; amp_gate_bound.sh
                         reruns it with the d_mlp scaling / robustness models from amp_mlp_width.py
    amp_gate_capacity.py (no run needed) planted gates: how many a width-H MLP can store (amp_gate_capacity.sh)
bash amp_checks.sh reruns all of them on the saved runs; logs go to downstream_amp/checks/.
"""
import argparse
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT_DIR.parent), str(ROOT_DIR), str(ROOT_DIR / "cross_token")]
import amp_models

import numpy as np
import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download
from scipy.sparse.csgraph import connected_components
from torch.func import grad_and_value, jvp, vmap
from torch.nn import functional as F

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--model", choices=amp_models.MODELS, default="swiglu",
                    help="swiglu: Elriggs/gpt2-swiglu-18l-9h-1152embd-v2; gpt2: GPT-2 small (GELU); "
                    "gpt2_noln: apollo-research/gpt2_noLN (GELU, LayerNorm fine-tuned out); "
                    "*_12l: Elriggs' 12-layer Muon models differing only in the MLP; gelu_1l / gelu_2l: NeelNanda/GELU_{1,2}L512W_C4_Code "
                    "(one / two layers: --layer 0 / 0-1)")
parser.add_argument("--layer", type=int, default=8, help="MLP whose input holds o and s")
parser.add_argument("--windows", type=int, default=480, help="fitting windows")
parser.add_argument("--heldout", type=int, default=256, help="held-out windows")
parser.add_argument("--ko", type=int, default=16, help="restrict o to the top-k principal directions (0: unrestricted)")
parser.add_argument("--band", default="", help="restrict o to principal directions a..b-1, e.g. 16,64 (overrides --ko)")
parser.add_argument("--pcs", default="", help="fix o to ± these principal directions (comma list); s is then the top "
                    "eigenvector of the contrast for each, and o is not fitted")
parser.add_argument("--half", type=int, default=-1, help="fit s on the first (0) or second (1) half of the fitting windows "
                    "(whitening and PCA from all of them), save, and stop: for split-half reliability")
parser.add_argument("--starts", type=int, default=32)
parser.add_argument("--iters", type=int, default=40)
parser.add_argument("--o-steps", type=int, default=30)
parser.add_argument("--tail", type=float, default=0.1, help="share of rows where o counts as on")
parser.add_argument("--draw", type=int, default=0, help="independent data draw (row group 60 + 2·draw) and seed")
parser.add_argument("--data", choices=["owt", "model", "bigram"], default="owt",
                    help="fitting and held-out windows: OpenWebText, samples from the model, or the model's bigram chain")
parser.add_argument("--report", type=int, default=6, help="attractors to check and describe")
args = parser.parse_args()

WIN, FIRST, TOPK = 32, 16, 512
P_ = WIN - FIRST
torch.set_float32_matmul_precision("high")
M = amp_models.load(args.model, WIN)
dev, D = M.dev, M.D
L = args.layer
MAX_L = getattr(M, "max_layer", M.n_layer - 2)
assert 0 <= L <= MAX_L, f"--layer must be in 0..{MAX_L} for {args.model}"
BOS, KEEP_BOS, SPECIAL = getattr(M, "bos", 50256), getattr(M, "keep_bos", False), getattr(M, "special", {50256})
tok = M.tokenizer
V = len(tok)
OUT = ROOT_DIR / "downstream_amp"
PREFIX = "" if args.model == "swiglu" else f"{args.model}_"
SUFFIX = "" if args.tail == 0.1 else f"_t{args.tail:g}"
OWT = hf_hub_download("apollo-research/Skylion007-openwebtext-tokenizer-gpt2", "data/train-00000-of-00073.parquet",
                      repo_type="dataset")


@torch.no_grad()
def model_ids(count, seed, batch=128):
    # temperature-1 samples from the model after <|endoftext|> (the model's own start token BOS, kept at position 0 if
    # KEEP_BOS); windows containing a special token are resampled
    g = torch.Generator(device=dev).manual_seed(seed)
    kept, total = [], 0
    special = torch.tensor(sorted(SPECIAL), device=dev)
    while total < count:
        seq = torch.full((batch, 1), BOS, device=dev)
        for _ in range(WIN - 1 if KEEP_BOS else WIN):
            seq = torch.cat([seq, torch.multinomial(torch.softmax(M.next_logits(seq), -1), 1, generator=g)], 1)
        seq = seq if KEEP_BOS else seq[:, 1:]
        seq = seq[~torch.isin(seq[:, 1:] if KEEP_BOS else seq, special).any(1)]
        kept.append(seq)
        total += len(seq)
    return torch.cat(kept)[:count]


def bigram_ids(count, seed):
    # Markov chain on the model's own bigram distribution (dct_etc/bigram/chain[_gpt2].pt: top-64 continuations per token)
    next_ids, next_probs = torch.load(M.bigram_chain())
    g = torch.Generator(device=dev).manual_seed(seed)
    token = torch.full((count,), BOS, device=dev)
    out = [token] if KEEP_BOS else []
    for _ in range(WIN - 1 if KEEP_BOS else WIN):
        token = next_ids[token, torch.multinomial(next_probs[token].float(), 1, generator=g)[:, 0]]
        out.append(token)
    return torch.stack(out, 1)


def windows(kind, split, count):
    # split 0: fitting windows of this draw; split 1: held-out windows (shared across draws)
    if kind == "owt":
        return owt(60 + 2 * args.draw, count, 2 * args.draw) if split == 0 else owt(61, count, 1)
    seed = 1000 + 2 * args.draw if split == 0 else 9001
    return model_ids(count, seed) if kind == "model" else bigram_ids(count, seed)


def owt(group, count, seed):
    table, rng, out = pq.ParquetFile(OWT).read_row_group(group), np.random.default_rng(seed), []
    retok = getattr(M, "retokenize", None)
    for i in range(table.num_rows):
        seq = np.asarray(table.column(0)[i].as_py())
        if retok is not None:
            # models with their own tokenizer: the text of a 2·WIN-token GPT-2 span, retokenised by the model's adapter
            if len(seq) > 2 * WIN + 8:
                o_ = int(rng.integers(0, len(seq) - 2 * WIN))
                if not (seq[o_:o_ + 2 * WIN] == 50256).any():
                    w = retok(seq[o_:o_ + 2 * WIN])
                    if w is not None:
                        out.append(np.asarray(w))
        elif len(seq) > WIN + 8:
            o_ = int(rng.integers(0, len(seq) - WIN))
            if not (seq[o_:o_ + WIN] == 50256).any():
                out.append(seq[o_:o_ + WIN])
        if len(out) == count:
            break
    return torch.tensor(np.stack(out), device=dev)


def downstream(x, r, ctx, theta):
    # MLP L with its (normalised) input shifted by theta, then the later blocks
    return M.downstream(x, r, ctx, theta, L)


def rows_of(ctx, sl):
    return tuple(c[sl] for c in ctx)


class Rows:
    @torch.no_grad()
    def __init__(self, ids):
        self.ids, self.n = ids, len(ids)
        self.x, self.r, self.ctx = M.capture(ids, L)
        ps, ixs, ents = [], [], []
        for s in range(0, self.n, 16):
            sl = slice(s, s + 16)
            lp = torch.log_softmax(M.logits(downstream(self.x[sl], self.r[sl], rows_of(self.ctx, sl), 0.0)[:, FIRST:]), -1)
            ents.append(-(lp.exp() * lp).sum(-1))
            p_, i_ = lp.exp().topk(TOPK, -1)
            ps.append(p_)
            ixs.append(i_)
        self.p, self.idx, self.ent = torch.cat(ps), torch.cat(ixs), torch.cat(ents)

    def z(self):
        return self.x[:, FIRST:]


def gain_fn(x, r, ctx, p, idx, push):
    C = jvp(lambda th: M.head(downstream(x, r, ctx, th)[:, FIRST:], idx), (torch.zeros_like(x),), (push.expand_as(x),))[1]
    # = ½(Σ p C² - (Σ p C)²) with p the top-k mass s ≤ 1, written without the cancellation that sends near-deterministic
    # rows to ≤ 0 (gpt2_noln, whose logits are not normalised)
    m = (p * C).sum(-1, keepdim=True)
    return 0.5 * ((p * (C - m).square()).sum(-1) + (1 - p.sum(-1)).clamp_min(0) * m[..., 0].square())


def step(s, w, x, r, ctx, p, idx):
    (g,), (_, e) = grad_and_value(lambda s: (lambda e: ((w * e).sum(), e))(gain_fn(x, r, ctx, p, idx, ROOT @ s)),
                                  argnums=(0,), has_aux=True)(s)
    return g, e


vstep = vmap(step, in_dims=(1, 2, None, None, None, None, None), out_dims=(1, 0), chunk_size=8)


def sweep(R, Sv, Wt=None, shift=None, batch=4):
    # gains (F, n, P_) of whitened pushes Sv (D, F); with row weights Wt (n, P_, F) also ∇ of Σ Wt·gain
    Wt = torch.zeros(R.n, P_, Sv.shape[1], device=dev) if Wt is None else Wt
    grad, G = torch.zeros_like(Sv), torch.zeros(Sv.shape[1], R.n, P_, device=dev)
    for s in range(0, R.n, batch):
        p_ = slice(s, s + batch)
        x = R.x[p_] + (0.0 if shift is None else shift)
        g, e = vstep(Sv, Wt[p_], x, R.r[p_], rows_of(R.ctx, p_), R.p[p_], R.idx[p_])
        grad += g.detach()
        G[:, p_] = e.detach()
    return grad, G


def tau(R):
    pr = F.normalize(torch.randn(D, 8, device=dev, generator=torch.Generator(device=dev).manual_seed(7)), dim=0)
    return sweep(R, pr)[1].mean(0)


def coords(R, O):
    return torch.einsum("npd,df->fnp", R.z() - MU, IROOT @ O)


def onoff(c):
    on = (c >= c.flatten(1).quantile(1 - args.tail, dim=1)[:, None, None]).float()
    return on / on.sum((1, 2), keepdim=True) - (1 - on) / (1 - on).sum((1, 2), keepdim=True)


def contrast(R, O, Sv, tau_):
    return (onoff(coords(R, O)) * sweep(R, Sv)[1] / tau_).sum((1, 2))


begin = time.time()
fit, held = Rows(windows(args.data, 0, args.windows)), Rows(windows(args.data, 1, args.heldout))
Zf = fit.z().reshape(-1, D).double()
MU = Zf.mean(0).float()
C = torch.cov(Zf.T)
C += 1e-3 * C.trace() / D * torch.eye(D, device=dev, dtype=C.dtype)
ev, evec = torch.linalg.eigh(C)
ROOT, IROOT = (evec @ torch.diag(ev.sqrt()) @ evec.T).float(), (evec @ torch.diag(ev.rsqrt()) @ evec.T).float()
tau_f, tau_h = tau(fit), tau(held)
if args.half >= 0:
    half = slice(0, fit.n // 2) if args.half == 0 else slice(fit.n // 2, fit.n)
    for a_ in ("ids", "x", "r", "p", "idx", "ent"):
        setattr(fit, a_, getattr(fit, a_)[half])
    fit.ctx, fit.n, tau_f = rows_of(fit.ctx, half), len(fit.ids), tau_f[half]
if args.band:
    a_, b_ = map(int, args.band.split(","))
    EK = evec.flip(1)[:, a_:b_].float()
else:
    EK = evec.flip(1)[:, :args.ko].float() if args.ko else torch.eye(D, device=dev)
print(f"{args.model} MLP {L}: {args.data} data, fit {fit.n}×{P_} rows (draw {args.draw}), held-out {held.n}×{P_}; o in {('PCs ' + args.band) if args.band else ('top ' + str(args.ko or D))} principal directions "
      f"; setup {time.time() - begin:.0f} s", flush=True)

g0 = torch.Generator(device=dev).manual_seed(args.draw)
picks = torch.randint(fit.n * P_, (2, args.starts), device=dev, generator=g0)
white = lambda z: F.normalize((z - MU) @ IROOT, dim=-1)
O = F.normalize(EK @ (EK.T @ white(fit.z().reshape(-1, D)[picks[0]]).T), dim=0).contiguous()
if args.pcs:
    idx_pc = [int(i) for i in args.pcs.split(",")]
    Epc = evec.flip(1)[:, idx_pc].float()
    O = torch.cat([Epc, -Epc], 1).contiguous()                 # in whitened coordinates a PC is its own read direction
    pc_names = [f"+PC{i}" for i in idx_pc] + [f"-PC{i}" for i in idx_pc]
    args.starts = O.shape[1]
    picks = torch.randint(fit.n * P_, (2, args.starts), device=dev, generator=g0)
Sv = white(fit.z().reshape(-1, D)[picks[1]]).T.contiguous()
perp = lambda Sv, O: F.normalize(Sv - (Sv * O).sum(0, keepdim=True) * O, dim=0)
Sv = perp(Sv, O)
sigma = None
for it in range(args.iters):
    W = (onoff(coords(fit, O)) / tau_f).permute(1, 2, 0)
    grad, G = sweep(fit, Sv, W)
    val = (W.permute(2, 0, 1) * G).sum((1, 2))
    sigma = grad.norm(dim=0) if sigma is None else sigma
    newS = perp(F.normalize(grad + sigma * Sv, dim=0), O)
    rel = (G / tau_f).detach()
    Op = (EK.T @ O).T.clone().requires_grad_()
    opt = torch.optim.Adam([Op], lr=0.01)
    for _ in range(0 if args.pcs else args.o_steps):
        o = F.normalize(Op @ EK.T, dim=-1)
        c = torch.einsum("npd,fd->fnp", fit.z() - MU, o @ IROOT.T)
        c = (c - c.mean((1, 2), keepdim=True)) / c.std((1, 2), keepdim=True)
        w = torch.sigmoid((c - c.detach().flatten(1).quantile(1 - args.tail, dim=1)[:, None, None]) / 0.1)
        loss = -((w * rel).sum((1, 2)) / w.sum((1, 2)) - ((1 - w) * rel).sum((1, 2)) / (1 - w).sum((1, 2))).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    newO = O if args.pcs else F.normalize(Op.detach() @ EK.T, dim=-1).T.contiguous()
    drift = torch.minimum((newS * Sv).sum(0).abs(), (newO * O).sum(0).abs())
    O, Sv = newO, perp(newS, newO)
    if (it + 1) % 10 == 0 or it == args.iters - 1:
        print(f"  iter {it + 1}: contrast median {val.median():.3g}, max {val.max():.3g}; min |cos| to previous "
              f"{drift.min():.3f}, median {drift.median():.4f} ({time.time() - begin:.0f} s)", flush=True)

if args.half >= 0:
    OUT.mkdir(exist_ok=True)
    path = OUT / f"{PREFIX}L{L}_{'PCs' if args.pcs else 'K' + str(args.ko)}_{args.data}_d{args.draw}_n{args.windows}{SUFFIX}_half{args.half}.pt"
    torch.save({"O": O.cpu(), "S": Sv.cpu(), "names": pc_names if args.pcs else None, "evec": evec.cpu(), "ev": ev.cpu(),
                "args": vars(args)}, path)
    print(f"saved {path} ({time.time() - begin:.0f} s)")
    sys.exit()
torch.set_grad_enabled(False)
score = contrast(fit, O, Sv, tau_f)
sim = torch.minimum((O.T @ O).abs(), (Sv.T @ Sv).abs()).cpu().numpy() > 0.8
label = connected_components(sim, directed=False)[1]
cnt = np.bincount(label)
reps = [int(np.nonzero(label == k)[0][np.argmax(score.cpu().numpy()[label == k])]) for k in np.argsort(-cnt) if cnt[k] >= 2]
print(f"\n{len(reps)} attractors (≥ 2 of {args.starts} starts), basins {sorted(cnt.tolist(), reverse=True)[:10]}")
if args.pcs:
    reps = list(range(args.starts))
    args.report = args.starts
elif not reps:
    reps = [int(i) for i in torch.argsort(score, descending=True)[:args.report]]
    print("no recurring end points; checking the highest-scoring end points instead")

R_ = F.normalize(torch.randn(D, 8, device=dev, generator=g0), dim=0)
null = contrast(held, R_, perp(F.normalize(torch.randn(D, 8, device=dev, generator=g0), dim=0), R_), tau_h)
print(f"held-out contrast of random (o, s ⟂ o) pairs: median {null.median():.3g}, max |·| {null.abs().max():.3g}")

STOP = set("""a an the and or but if then else of to in on at by for with from as into onto upon about above below over under
between through during before after since until while than that this these those there here what which who whom whose when
where why how all any both each few more most other some such no nor not only own same so too very can will just should
would could may might must shall do does did done doing have has had having be been being is are was were am i me my we
our you your he him his she her it its they them their also however even still yet now well much many one two said""".split())
dec = [tok.decode([i]) for i in range(V)]
content = torch.tensor([d.strip().isalpha() and len(d.strip()) >= 3 and d.strip().lower() not in STOP for d in dec], device=dev)
nxt = torch.cat([held.ids[:, FIRST + 1:], held.ids[:, -1:]], 1)
next_content, valid = content[nxt], torch.ones_like(nxt, dtype=torch.bool)
valid[:, -1] = False


def ctx(R, row, before=6):
    w, p = divmod(int(row), P_)
    p += FIRST
    return (tok.decode(R.ids[w, max(0, p - before):p]) + "[" + tok.decode(R.ids[w, p:p + 1]) + "]").replace("\n", "⏎")


def fires(R, coord, k=6):
    out = []
    for sign in (1, -1):
        rows, docs = [], set()
        for r in (sign * coord.flatten()).argsort(descending=True).tolist():
            if r // P_ not in docs:
                docs.add(r // P_)
                rows.append(r)
            if len(rows) == k:
                break
        out.append(" | ".join(ctx(R, r) for r in rows))
    return out


def opened_channel(R, s, on):
    acc_on, acc_off = torch.zeros(V, device=dev), torch.zeros(V, device=dev)
    for s0 in range(0, R.n, 16):
        sl = slice(s0, s0 + 16)
        f = lambda th: torch.log_softmax(M.logits(downstream(R.x[sl], R.r[sl], rows_of(R.ctx, sl), th)[:, FIRST:])[..., :V], -1)
        _, dlp = jvp(f, (torch.zeros_like(R.x[sl]),), ((ROOT @ s).expand_as(R.x[sl]),))
        m = on[sl].float()[..., None]
        acc_on += (dlp * m).sum((0, 1))
        acc_off += (dlp * (1 - m)).sum((0, 1))
    eff = acc_on / on.sum() - acc_off / (~on).sum()
    show = lambda v: " ".join(repr(dec[i]) for i in v.tolist())
    return show(eff.topk(10).indices), show((-eff).topk(10).indices)


def within(rel, on, strata):
    vals = []
    for sel in strata:
        a, b = sel & on, sel & ~on
        if a.sum() >= 20 and b.sum() >= 20:
            vals.append(((rel[a].mean() - rel[b].mean()).item(), int(sel.sum())))
    return sum(v * w for v, w in vals) / sum(w for _, w in vals) if vals else float("nan")


pr = F.normalize(torch.randn(D, 8, device=dev, generator=torch.Generator(device=dev).manual_seed(99)), dim=0)
_, G_rand = sweep(held, pr)
checks = []
for k, j in enumerate(reps[:args.report]):
    o, s = O[:, j], Sv[:, j]
    c = coords(held, o[:, None])[0]
    on = c >= c.flatten().quantile(1 - args.tail)
    delta = (c[on].mean() - c[~on].mean()).item()
    _, Gb = sweep(held, s[:, None])
    _, Gp = sweep(held, s[:, None], shift=delta * (ROOT @ o))
    rnd = torch.randn(D, device=dev, generator=g0)
    rnd = rnd - (rnd @ (IROOT @ o)) / (IROOT @ o).square().sum() * (IROOT @ o)
    _, Gr = sweep(held, s[:, None], shift=F.normalize(rnd, dim=0) * (delta * (ROOT @ o)).norm())
    _, Gp_rand = sweep(held, pr, shift=delta * (ROOT @ o))
    rel = Gb[0] / tau_h
    obs = (rel[on].mean() - rel[~on].mean()).item()
    caus = ((Gp - Gb)[0][~on] / tau_h[~on]).mean().item() / obs
    caus_r = ((Gr - Gb)[0][~on] / tau_h[~on]).mean().item() / obs
    spec_s = (Gp / Gb.clamp_min(1e-12))[0][~on].mean().item()
    spec_r = (Gp_rand / G_rand.clamp_min(1e-12))[:, ~on].mean(1).median().item()
    q = held.ent.quantile(torch.linspace(0, 1, 6, device=dev))
    ent_ratio = within(rel, on, [(held.ent >= q[i]) & (held.ent <= q[i + 1]) for i in range(5)]) / obs
    cont_ratio = within(rel, on, [valid & next_content, valid & ~next_content]) / obs
    promotes, suppresses = (None, None) if args.pcs else opened_channel(held, s, on)
    checks.append(dict(contrast=obs, causal=caus, causal_random=caus_r, gain_s=spec_s, gain_random=spec_r,
                       entropy=ent_ratio, content=cont_ratio))
    if args.pcs:
        _, Gn = sweep(held, perp(F.normalize(torch.randn(D, 8, device=dev, generator=g0), dim=0), o[:, None].expand(D, 8)))
        nul = ((Gn / tau_h)[:, on].mean(1) - (Gn / tau_h)[:, ~on].mean(1)).abs().max().item()
        checks[-1].update(name=pc_names[j], null=nul, fit=score[j].item())
        print(f"  {pc_names[j]:>7s}: fit {score[j]:6.3g} held-out {obs:6.3g} (random s, same o: max {nul:.3g}) | causal {caus:5.2f} "
              f"(random {caus_r:5.2f}) | gain own s ×{spec_s:.2f} vs random ×{spec_r:.2f} | nulls {ent_ratio:.2f} {cont_ratio:.2f}",
              flush=True)
        continue
    print(f"\nattractor {k} ({cnt[label[j]]} starts): contrast fit {score[j]:.3g}, held-out {obs:.3g}")
    print(f"  causal: pushing o on reproduces {caus:.2f} of the contrast (random push {caus_r:.2f}); gain of s ×{spec_s:.2f}, "
          f"of random s ×{spec_r:.2f}")
    print(f"  nulls (within / overall): next-token entropy quintiles {ent_ratio:.2f}, next token content word {cont_ratio:.2f}")
    for name, v in (("o", o), ("s", s)):
        a, b = fires(held, coords(held, v[:, None])[0])
        print(f"  {name} + {a}\n    − {b}")
    print(f"  opened channel promotes {promotes}\n                 suppresses {suppresses}", flush=True)

OUT.mkdir(exist_ok=True)
tag = "PCs" if args.pcs else (f"B{args.band.replace(',', '-')}" if args.band else f"K{args.ko}")
path = OUT / f"{PREFIX}L{L}_{tag}_{args.data}_d{args.draw}_n{args.windows}{SUFFIX}.pt"
torch.save({"O": O.cpu(), "S": Sv.cpu(), "score": score.cpu(), "label": label, "reps": reps, "checks": checks,
            "ROOT": ROOT.cpu(), "IROOT": IROOT.cpu(), "MU": MU.cpu(), "args": vars(args)}, path)
print(f"\nsaved {path} ({time.time() - begin:.0f} s)")
