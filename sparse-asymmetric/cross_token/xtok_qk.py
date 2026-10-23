"""Query/key-split cross-token DCT: gated interactions between a key-side feature at earlier tokens and a query-side
feature at the current token.

Sites: l is pushed only into the KEY input of block S's attention at the earlier tokens 1..t-1, r only into the QUERY input
at the current token t (values and everything else untouched), so l and r meet in each head's query-key form. Objective
(no ablation): l's first-order Fisher gain at the next-token distribution with r pushed by R versus not pushed, in the same
context, relative to the gain of random pushes at that row (τ):
    gain(l | θ_q) = ½ Var_{p_t}( ∂_α logits_t[ key input + α·P l at earlier tokens, query input + θ_q at t ] )
    J(l, r)       = Σ_rows [gain(l | R·sd_r·r̂) - gain(l | 0)] / τ_row
The push on r is R natural standard deviations of the fitting rows along r's own direction (default R = 1.5). In whitened
units instead (--whitened-push), the objective prefers low-variance directions where a fixed push is many natural SDs, i.e.
gates that natural text never operates. P = Σ^½ with a Ledoit-Wolf shrinkage covariance of the fitting rows only.
l: shifted power step; r: normalised gradient step; census of independent random starts.

Evaluation (ablations only here), held-out model samples and WikiText: l's natural key-side component is removed at the
earlier tokens with r's natural query-side component present vs removed at t. Removing r moves it to its mean, which
closes a +r gate where r is naturally high and opens it where r is low, so the ratio is reported split by r's natural
value; a gate shows ratio << 1 for high r and >> 1 for low r, against principal-component controls. Also reported: the
share of the interaction carried by single heads, and the match with each head's top whitened query-key singular pairs.

Usage: python xtok_qk.py [--S 12] [--windows 64] [--R 1.5] [--starts 32] [--iters 30] [--seed 0] [--whitened-push]
Outputs: cross_token/qk_S{S}_R{R}_seed{seed}.pt
"""
import argparse
import sys
import time

sys.path.insert(0, "/home/aj/redesigned-octo-couscous/sparse-asymmetric")
_argv, sys.argv = sys.argv, [sys.argv[0]]
import xtok as X
sys.argv = _argv

import numpy as np
import torch
from scipy.sparse.csgraph import connected_components
from torch.func import grad_and_value, jvp, vmap
from torch.nn import functional as F

parser = argparse.ArgumentParser()
parser.add_argument("--S", type=int, default=12)
parser.add_argument("--windows", type=int, default=64)
parser.add_argument("--R", type=float, default=1.5, help="push on r in natural SDs along r")
parser.add_argument("--starts", type=int, default=32)
parser.add_argument("--iters", type=int, default=30)
parser.add_argument("--probes", type=int, default=8)
parser.add_argument("--chunk", type=int, default=8)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--whitened-push", action="store_true", help="push r by R in whitened units instead of natural SDs")
args = parser.parse_args()
torch.set_float32_matmul_precision("high")
dev, D, S, H, hd, topk, T = X.device, X.D, args.S, X.H, X.hd, 512, X.readouts
X.out.mkdir(exist_ok=True)
blk, nxt = X.blocks[S], X.blocks[S + 1]
begin = time.time()


def attn_split(attn, x, first, tq, tk, qmask=None, kmask=None):
    B, L_, _ = x.shape
    view = lambda z: z.view(B, L_, H, hd)
    q0, k0 = view(attn.c_q(x)), view(attn.c_k(x))
    dq, dk = view(attn.c_q(tq.expand_as(x))), view(attn.c_k(tk.expand_as(x)))
    if qmask is not None:
        dq = dq * qmask[None, None, :, None]
    if kmask is not None:
        dk = dk * kmask[None, None, :, None]
    q, k, v = q0 + dq, k0 + dk, view(attn.c_v(x))
    v = (1 - attn.lamb) * v + attn.lamb * first.view_as(v)
    cos, sin = X.cos_cached[None, :L_, None, :], X.sin_cached[None, :L_, None, :]
    q = X.apply_rotary_emb(F.rms_norm(q, (hd,)), cos, sin).transpose(1, 2)
    k = X.apply_rotary_emb(F.rms_norm(k, (hd,)), cos, sin).transpose(1, 2)
    causal = torch.ones(L_, L_, device=x.device, dtype=torch.bool).tril()
    pattern = (q @ k.transpose(-1, -2) / hd ** 0.5).masked_fill(~causal, float("-inf")).softmax(-1)
    return attn.c_proj((pattern @ v.transpose(1, 2)).transpose(1, 2).reshape(B, L_, D))


def forward(m, ini, fir, tq, tk, qmask=None, kmask=None):
    # block S with split query/key pushes on its attention input, then blocks S+1..17; returns the final residual
    x = m + attn_split(blk.attn, F.rms_norm(m, (D,)), fir, tq, tk, qmask, kmask)
    x = x + X.mlp(blk.mlp, F.rms_norm(x, (D,)))
    x = nxt.lambdas[0] * x + nxt.lambdas[1] * ini
    return X.run(x, ini, fir, S + 1, X.n_layer)


def fisher_head(aux, t):
    p, W = aux[:, :topk], X.lm_weight[aux[:, topk:].long()].float()
    return p, lambda h: 30 * torch.tanh(torch.einsum("bd,bkd->bk", F.rms_norm(h[:, t], (D,)), W) / 30)


def gain(l, tq, m, ini, fir, aux, t, qmask=None, kmask=None):
    earlier, _ = X.masks(t)
    p, head = fisher_head(aux, t)
    zero = torch.zeros_like(m)
    C = jvp(lambda tk: head(forward(m, ini, fir, tq, tk, qmask, kmask)), (zero,), ((earlier * l).expand_as(m),))[1]
    return 0.5 * ((p * C.square()).sum(-1) - (p * C).sum(-1).square())


class Rows:
    def __init__(self, ids):
        r = X.Rows(ids, S, "fisher", X.n_layer, topk)
        self.ids, self.m, self.ini, self.fir, self.aux, self.n = ids, r.m, r.initial, r.first, r.aux, r.count
        self.xhat = F.rms_norm(self.m, (D,))


def shrinkage_sqrt(xhat):
    raw = xhat[:, 1:].reshape(-1, D)
    mean = raw.mean(0)
    Z = (raw - mean).double()
    n = len(Z)
    Sm = Z.T @ Z / n
    eye = torch.eye(D, device=dev, dtype=Sm.dtype)
    mu_ = Sm.trace() / D
    alpha = ((Z.square().sum(1).square().sum() / n ** 2 - Sm.square().sum() / n).clamp_min(0)
             / (Sm - mu_ * eye).square().sum()).clamp(max=1.0)
    ev, evec = torch.linalg.eigh(alpha * mu_ * eye + (1 - alpha) * Sm)
    P = (evec * ev.clamp_min(0).sqrt()) @ evec.T
    return (P / P.diag().mean().sqrt()).float(), alpha.item(), n, evec.flip(1)[:, :32].T.float(), mean


train = Rows(X.sample_ids("model", 300 + 1000 * args.seed, args.windows))
P, alpha, nvec, pcs, mu = shrinkage_sqrt(train.xhat)
with torch.no_grad():
    check = (forward(train.m[:8], train.ini[:8], train.fir[:8], torch.zeros(1, 1, D, device=dev), torch.zeros(1, 1, D, device=dev))
             - X.run(train.m[:8], train.ini[:8], train.fir[:8], S, X.n_layer)).abs().max().item()
print(f"S={S}, seed {args.seed}: {args.windows} model windows ({train.n * len(T)} rows); covariance from these rows only ({nvec} "
      f"vectors, shrinkage {alpha:.2f}); R = {args.R:g}; forward check max |diff| {check:.1e}", flush=True)


def make_step(t):
    _, here = X.masks(t)

    def step(l, r, w, m, ini, fir, aux):
        def objective(l, r):
            on = gain(P @ l, here * r_push(r), m, ini, fir, aux, t)
            off = gain(P @ l, torch.zeros(1, 1, D, device=dev), m, ini, fir, aux, t)
            return (w * (on - off)).sum(), (on, off)
        (gl, gr), (_, (on, off)) = grad_and_value(objective, argnums=(0, 1), has_aux=True)(l, r)
        return gl, gr, on, off

    return vmap(step, in_dims=(1, 1, None, None, None, None, None), out_dims=(1, 1, 0, 0), chunk_size=args.chunk)


Xc_train = train.xhat[:, 1:].reshape(-1, D) - mu


def r_push(r):
    if args.whitened_push:
        return args.R * (P @ r)
    rd = F.normalize(P @ r, dim=0)
    return args.R * (Xc_train @ rd).std() * rd


sd = lambda v: (Xc_train @ v).std()
steps = {t: make_step(t) for t in T}


def sweep(rows, L, R, w, batch=32):
    # w: (T, n) row weights; returns ∇l, ∇r of Σ w (on - off), and per-row on/off gains (F, T, n)
    gL, gR = torch.zeros_like(L), torch.zeros_like(R)
    on = torch.zeros(L.shape[1], len(T), rows.n, device=dev)
    off = torch.zeros_like(on)
    for ti, t in enumerate(T):
        for s in range(0, rows.n, batch):
            part = slice(s, s + batch)
            a, b, c, d = steps[t](L, R, w[ti, part], rows.m[part, :t + 1], rows.ini[part, :t + 1], rows.fir[part, :t + 1],
                                  rows.aux[t][part])
            gL += a.detach()
            gR += b.detach()
            on[:, ti, part], off[:, ti, part] = c.detach(), d.detach()
    return gL, gR, on, off


def tau(rows):
    probes = F.normalize(torch.randn(D, args.probes, device=dev, generator=torch.Generator(device=dev).manual_seed(77)), dim=0)
    return sweep(rows, probes, probes, torch.zeros(len(T), rows.n, device=dev))[3].mean(0)


tau_train = tau(train)
w_train = 1 / tau_train / (train.n * len(T))
gen = torch.Generator(device=dev).manual_seed(args.seed)
L, R = (F.normalize(torch.randn(D, args.starts, device=dev, generator=gen), dim=0) for _ in range(2))
cl = torch.ones(args.starts, device=dev)
cr = torch.full((args.starts,), 2.0, device=dev)
previous = torch.full((args.starts,), -float("inf"), device=dev)
for it in range(args.iters):
    gL, gR, on, off = sweep(train, L, R, w_train)
    value = ((on - off) * w_train).sum((1, 2))
    worse = value < previous
    cl, cr = torch.where(worse, cl * 2, (cl * 0.8).clamp_min(0.5)), torch.where(worse, cr * 2, (cr * 0.8).clamp_min(0.5))
    previous = value
    newL = F.normalize(gL + cl * gL.norm(dim=0) * L, dim=0)
    newR = F.normalize(gR + cr * gR.norm(dim=0) * R, dim=0)
    drift = torch.minimum((newL * L).sum(0).abs(), (newR * R).sum(0).abs())
    L, R = newL, newR
    if it % 5 == 4 or it == args.iters - 1:
        print(f"  sweep {it + 1}: objective (contrast / τ, per row) median {value.median().item():.3f}, max {value.max().item():.3f}; "
              f"min |cos| to previous {drift.min().item():.4f}, median {drift.median().item():.5f} ({time.time() - begin:.0f} s)",
              flush=True)

pl, pr = F.normalize((P @ L).T, dim=1), F.normalize((P @ R).T, dim=1)       # pushed directions in attention-input space
sim = ((pl @ pl.T).abs() * (pr @ pr.T).abs()).cpu().numpy()
label = connected_components(sim > 0.8, directed=False)[1]
counts = np.bincount(label)
reps = [int(np.nonzero(label == k)[0][0]) for k in np.argsort(-counts)]
print(f"\n{sum(counts >= 2)} attractors from {args.starts} starts, basin sizes {sorted(counts.tolist(), reverse=True)[:10]}")
chosen = reps[:4]

# ---------------------------------------------------------------- evaluation
torch.set_float32_matmul_precision("highest")
held = Rows(X.sample_ids("model", 5000 + 1000 * args.seed, 128))
wiki = Rows(X.wikitext_ids(0, 128))
tau_held = tau(held)
gen = torch.Generator(device=dev).manual_seed(5)
rand = [(F.normalize(P @ torch.randn(D, device=dev, generator=gen), dim=0), F.normalize(P @ torch.randn(D, device=dev, generator=gen), dim=0))
        for _ in range(4)]
with torch.no_grad():
    rL, rR = torch.stack([a for a, _ in rand], 1), torch.stack([b for _, b in rand], 1)
    Pinv = torch.linalg.pinv(P)
    _, _, on_f, off_f = sweep(held, L[:, chosen], R[:, chosen], torch.zeros(len(T), held.n, device=dev))
    _, _, on_n, off_n = sweep(held, Pinv @ rL, Pinv @ rR, torch.zeros(len(T), held.n, device=dev))
contrast = lambda on, off: ((on - off) / tau_held).mean((1, 2))
print(f"held-out objective (contrast / τ, per row): end points {' '.join(f'{x:.3f}' for x in contrast(on_f, off_f).tolist())} | "
      f"random pairs {' '.join(f'{x:.3f}' for x in contrast(on_n, off_n).tolist())}")


@torch.no_grad()
def gating(rows, l, r, push_vec):
    # natural key-side component of l (earlier tokens), natural query-side component of r (current token); KL at t
    out = []
    zero = torch.zeros(1, 1, D, device=dev)
    for t in T:
        earlier, here = X.masks(t)
        for s in range(0, rows.n, 64):
            part = slice(s, s + 64)
            m, ini, fir, xh = rows.m[part, :t + 1], rows.ini[part, :t + 1], rows.fir[part, :t + 1], rows.xhat[part, :t + 1]
            a = -earlier * ((xh - mu) @ l)[..., None] * l
            b = -here * ((xh - mu) @ r)[..., None] * r
            push = here * push_vec
            lp = lambda tq, tk: torch.log_softmax(X.logits(forward(m, ini, fir, tq, tk)[:, t]), -1)
            base, no_l = lp(zero, zero), lp(zero, a)
            no_r, no_lr = lp(b, zero), lp(b, a)
            pushed, pushed_no_l = lp(push, zero), lp(push, a)
            kl = lambda p, q: (p.exp() * (p - q)).sum(-1)
            z = ((xh[:, t] - mu) @ r) / sd(r)
            out.append(torch.stack([kl(base, no_l), kl(no_r, no_lr), kl(pushed, pushed_no_l), z], 1))
    return torch.cat(out)


@torch.no_grad()
def head_share(rows, l, r, n=48):
    # contrast with both pushes restricted to one head, for each head
    shares = []
    for h in range(H):
        mask = F.one_hot(torch.tensor(h, device=dev), H).float()
        tot = 0.0
        for t in T:
            _, here = X.masks(t)
            m, ini, fir, aux = rows.m[:n, :t + 1], rows.ini[:n, :t + 1], rows.fir[:n, :t + 1], rows.aux[t][:n]
            on = gain(l, here * r_push(torch.linalg.pinv(P) @ r), m, ini, fir, aux, t, mask, mask)
            off = gain(l, torch.zeros(1, 1, D, device=dev), m, ini, fir, aux, t, mask, mask)
            tot += ((on - off) / tau_held[T.index(t), :n]).mean().item()
        shares.append(tot)
    s = np.array(shares)
    return s, np.abs(s).max() / np.abs(s).sum()


Wq, Wk = blk.attn.c_q.weight.float(), blk.attn.c_k.weight.float()
qk_pairs = []
for h in range(H):
    M = P @ Wq[h * hd:(h + 1) * hd].T @ Wk[h * hd:(h + 1) * hd] @ P                # whitened query-key form of head h
    U_, s_, Vh = torch.linalg.svd(M)
    qk_pairs.append((U_[:, :4], Vh[:4]))

print("\nnatural-amplitude gating (KL at the next-token distribution): l's effect with r's natural component removed / present, "
      "split by r's natural value (z in natural SDs); a +r gate: << 1 for high r, >> 1 for low r")
print("  name | data | rows where l matters (KL > 0.001) | r high (z > 0.5) | r near mean | r low (z < -0.5) | "
      "l's effect, r pushed +R / not pushed")
candidates = [(f"end point {k} (basin {counts[label[j]]})", pl[j], pr[j], r_push(R[:, j])) for k, j in enumerate(chosen)]
candidates += [(f"random pair {k}", a, b, args.R * sd(b) * b) for k, (a, b) in enumerate(rand[:2])]
candidates += [(f"PC pair (key: PC {k}, query: PC {k + 1})", pcs[k], pcs[k + 1], args.R * sd(pcs[k + 1]) * pcs[k + 1]) for k in (0, 2)]
print("  push sizes in data SDs along r: end points " + " ".join(f"{(c[3].norm() / sd(c[2])).item():.1f}" for c in candidates[:len(chosen)]))
results = {}
for name, l, r, push_vec in candidates:
    for data_name, rows in (("model held-out", held), ("WikiText", wiki)):
        k = gating(rows, l, r, push_vec)
        act = k[:, 0] > 0.001
        results[name, data_name] = k.cpu()
        cells = []
        for sel in (act & (k[:, 3] > 0.5), act & (k[:, 3].abs() <= 0.5), act & (k[:, 3] < -0.5)):
            cells.append(f"{(k[sel, 1].mean() / k[sel, 0].mean()).item():.2f} on {int(sel.sum())}" if sel.any() else "-")
        pushed = (k[act, 2].mean() / k[act, 0].mean()).item() if act.any() else float("nan")
        print(f"  {name:36s} {data_name:15s} {int(act.sum()):4d} of {len(k)} | {' | '.join(cells)} | {pushed:.2f}", flush=True)

print("\nwhere the interaction sits")
for k, j in enumerate(chosen[:3]):
    shares, top = head_share(held, P @ L[:, j], P @ R[:, j])
    best = int(np.argmax(np.abs(shares)))
    align = max(((qk_pairs[h][0].T @ F.normalize(R[:, j], dim=0)).abs().max() * (qk_pairs[h][1] @ F.normalize(L[:, j], dim=0)).abs().max()).item()
                for h in range(H))
    print(f"  end point {k}: largest single-head share of the contrast {top:.2f} (head {S}.{best}); per-head contrast "
          f"{' '.join(f'{x:+.2f}' for x in shares)}; best match with a head's top-4 whitened QK singular pair (|cos q|·|cos k|) "
          f"{align:.2f}; share of l / r in the top 32 PCs {((pl[j] @ pcs.T) ** 2).sum().item():.2f} / {((pr[j] @ pcs.T) ** 2).sum().item():.2f}",
          flush=True)

dev_held = (held.xhat - mu)[:, 1:]
for k, j in enumerate(chosen[:3]):
    print(f"\nend point {k} (basin {counts[label[j]]})")
    for side, v in (("l (key side, earlier tokens)", pl[j]), ("r (query side, current token)", pr[j])):
        a, b = X.fires_on(held.ids, dev_held, v, 4)
        print(f"  {side}: {a}\n  {'':29s}  {b}")
    kk = results[f"end point {k} (basin {counts[label[j]]})", "model held-out"]
    for i in torch.argsort(kk[:, 0] - kk[:, 1], descending=True)[:3].tolist():
        ti, w = divmod(i, held.n)
        print(f"    l's effect {kk[i, 0]:.3f} with r, {kk[i, 1]:.3f} without: {X.context(held.ids, w, T[ti], 8)}")
torch.save({"L": pl.cpu(), "R": pr.cpu(), "Lw": L.cpu(), "Rw": R.cpu(), "P": P.cpu(), "label": label, "reps": reps,
            "results": results, "args": vars(args)},
           X.out / f"qk_S{S}_R{args.R:g}{'w' if args.whitened_push else ''}_seed{args.seed}.pt")
print(f"\ntotal {time.time() - begin:.0f} s")
