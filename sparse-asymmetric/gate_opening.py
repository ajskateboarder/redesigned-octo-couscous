"""Cross-gating between a run's attractors: does opener o_j also open attractor i's channel s_i?

    A(o_j, s_i)   held-out contrast of s_i (projected ⟂ o_j) between o_j's on-rows (top --tail) and the rest
Raw cross contrasts mislead, so each distinct pair (|cos o|, |cos s| < 0.5) is also split into
    overlap       the part implied by tail overlap alone: if s_i's channel depended only on o_i, then
                  A(o_j, s_i) = A_ii · (P(I | J) - P(I | not J)), I and J the on-sets of o_i and o_j
                  (two openers that are never on together give about -A_ii/9 with no mechanism at all)
    direct        effect of o_j away from o_i: mean gain of s_i on J \\ I minus on neither, on g/τ and on raw gain g
                  (τ, the per-row normaliser, differs between on-sets and can manufacture a difference by itself)

Usage: python gate_opening.py <run .pt in downstream_amp/ or a path>
Output: downstream_amp/checks/gate_opening_<run>.log
"""
import sys

import torch
from torch.nn import functional as F

from amp_common import from_run, log_to, openers

path, res, S_ = from_run(sys.argv[1])
log_to(f"gate_opening_{path.stem}")
dev, held, tau_h, coords, sweep, contrast, tail = S_.dev, S_.held, S_.tau_h, S_.coords, S_.sweep, S_.contrast, S_.args.tail
torch.set_grad_enabled(False)
reps, _ = openers(res)
k = len(reps)
O, S = res["O"][:, reps].to(dev), res["S"][:, reps].to(dev)
print(f"{S_.args.model} L{S_.L} ({path.name}): {k} attractors")

A = torch.zeros(k, k)
for i in range(k):
    Oi = O[:, i:i + 1].expand(-1, k).contiguous()
    A[i] = contrast(held, Oi, F.normalize(S - (S * Oi).sum(0, keepdim=True) * Oi, dim=0), tau_h).cpu()
torch.set_printoptions(precision=2, sci_mode=False, linewidth=200)
print("A(o_i, s_j), held-out (rows: opener, columns: channel):\n", A)
print("|cos o_i, o_j|:\n", (O.T @ O).abs().cpu())
print("|cos s_i, s_j|:\n", (S.T @ S).abs().cpu())

c = coords(held, O).flatten(1)
on = c >= c.quantile(1 - tail, dim=1, keepdim=True)
G = sweep(held, S)[1]
rel, raw = (G / tau_h).flatten(1), (G / tau_h.median()).flatten(1)
co, cs = (O.T @ O).abs(), (S.T @ S).abs()
rows = []
for i in range(k):
    I = on[i]
    Aii, Aii_raw = rel[i][I].mean() - rel[i][~I].mean(), raw[i][I].mean() - raw[i][~I].mean()
    for j in range(k):
        if j == i or co[i, j] >= 0.5 or cs[i, j] >= 0.5:
            continue
        J = on[j]
        rows.append((((I & J).sum() / J.sum()).item(),
                     ((rel[i][J].mean() - rel[i][~J].mean()) / Aii).item(),
                     ((I & J).sum() / J.sum() - (I & ~J).sum() / (~J).sum()).item(),
                     ((rel[i][J & ~I].mean() - rel[i][~J & ~I].mean()) / Aii).item(),
                     ((raw[i][J & ~I].mean() - raw[i][~J & ~I].mean()) / Aii_raw).item()))
if not rows:
    print("no distinct pairs")
    sys.exit()
t = torch.tensor(rows)
med = t.median(0).values
share = lambda v: f"> +0.25: {(v > .25).float().mean():.2f}, < -0.1: {(v < -.1).float().mean():.2f}, |·| < 0.1: {(v.abs() < .1).float().mean():.2f}"
print(f"\n{len(rows)} distinct ordered pairs (|cos o|, |cos s| < 0.5); all ratios relative to the channel's own contrast A_ii")
print(f"  tail overlap P(I | J): median {med[0]:.3f} (independent openers: {tail:.3f})")
print(f"  A(o_j, s_i) / A_ii:          median {med[1]:+.3f} | {share(t[:, 1])}")
print(f"  implied by overlap alone:    median {med[2]:+.3f}")
print(f"  direct, g/τ:                 median {med[3]:+.3f} | {share(t[:, 3])}")
print(f"  direct, raw gain:            median {med[4]:+.3f} | {share(t[:, 4])}")
