"""Where a run's openers sit in the PC spectrum and what their coordinate looks like on held-out rows.

    energy    share of each opener (a unit vector in whitened coordinates) in PC bands 0-16 / 16-64 / 64-256 / 256-end;
              a random direction has band size / D
    shape     of the held-out coordinate c(o): |skew| (Gaussian 0), bimodality coefficient BC = (skew² + 1) / kurtosis
              (Gaussian 0.33, two equal separated clumps up to 1; above ~0.55 suggests clumps), and the on/off gap:
              mean on-rows minus mean off-rows, in units of the off-rows' spread
    reference the same for random directions in the top 16 PCs and in all directions, and for each of the top 16 PCs
    scale     how large the model writes the opener's on-cluster, in raw (unwhitened) MLP-input units. Δ = mean z on
              on-rows minus off-rows; λ(Δ) = variance of z along Δ's direction and its rank among the PC variances
              (where that direction would sit as a PC); cluster share = tail(1 - tail)|Δ|² / λ(Δ), the part of that
              variance the on/off split explains on its own; |Δ| in units of the typical PC spread (√ median PC variance);
              coordinate PR = participation ratio of Δ over the D coordinates of z (1: a single coordinate, random ≈ D/3)
    --compare best |correlation| over this run's held-out rows between each opener's coordinate and the coordinates of
              another run's openers (each computed with its own run's mean and whitening, so runs fitted on different
              windows are comparable; for runs on the same windows it equals |cos| in whitened coordinates)

Usage: python amp_opener_shape.py <run .pt in downstream_amp/ or a path> [--compare <other run .pt>]
Output: downstream_amp/checks/amp_opener_shape_<run>.log
"""
import argparse

import torch
from torch.nn import functional as F

from amp_common import BEFORE_TAU, from_run, load_run, log_to, openers

parser = argparse.ArgumentParser()
parser.add_argument("run")
parser.add_argument("--compare", default="")
cli = parser.parse_args()
path, res, S_ = from_run(cli.run, stop=BEFORE_TAU)
log_to(f"amp_opener_shape_{path.stem}")
dev, held, coords, D, tail = S_.dev, S_.held, S_.coords, S_.D, S_.args.tail
torch.set_grad_enabled(False)
E = S_.evec.flip(1).float()                                      # PCs, descending variance
reps, rec = openers(res)
O = res["O"][:, reps].to(dev)


def shape(Od):
    c = coords(held, Od).flatten(1).double()
    c = (c - c.mean(1, keepdim=True)) / c.std(1, keepdim=True)
    sk, ku = (c ** 3).mean(1), (c ** 4).mean(1)
    on = c >= c.quantile(1 - tail, dim=1, keepdim=True)
    gap = torch.stack([(ci[oi].mean() - ci[~oi].mean()) / ci[~oi].std() for ci, oi in zip(c, on)])
    return sk.abs(), (sk ** 2 + 1) / ku, gap


bands = [(0, 16), (16, 64), (64, 256), (256, D)]
fmt = lambda v, n=2: "[" + ", ".join(f"{x:.{n}f}" for x in v.tolist()) + "]"
print(f"{S_.args.model} L{S_.L} ({path.name}): {len(reps)} openers, recurring: {rec}")
energy = (E.T @ O).square()
for a, b in bands:
    print(f"  energy in PCs {a}-{b}: {fmt(energy[a:b].sum(0))} (random direction {(b - a) / D:.2f})")
sk, bc, gap = shape(O)
print(f"  |skew| {fmt(sk, 1)} | BC {fmt(bc)} | on/off gap {fmt(gap, 1)}")
gen = torch.Generator(device=dev).manual_seed(3)
for name, basis in (("random in top-16 PCs", E[:, :16]), ("random in all directions", torch.eye(D, device=dev))):
    sk_r, bc_r, gap_r = shape(F.normalize(basis @ torch.randn(basis.shape[1], 64, device=dev, generator=gen), dim=0))
    print(f"  reference, {name} (64): |skew| median {sk_r.median():.1f} | BC median {bc_r.median():.2f} "
          f"(90th pct {bc_r.quantile(.9):.2f}) | gap median {gap_r.median():.1f} (90th pct {gap_r.quantile(.9):.1f})")
sk_p, bc_p, _ = shape(E[:, :16])
print(f"  reference, top-16 PCs individually: |skew| {fmt(sk_p, 1)} | BC {fmt(bc_p)}")
zh = held.z().reshape(-1, D)
C, ev = S_.C.float(), S_.ev.float()
print(f"  scale (raw MLP-input units; {D} PCs, √median PC variance {ev.median().sqrt():.3g}, √top {ev.max().sqrt():.3g}):")
c_h = coords(held, O).flatten(1)
for j in range(len(reps)):
    on = c_h[j] >= c_h[j].quantile(1 - tail)
    delta = zh[on].mean(0) - zh[~on].mean(0)
    u = F.normalize(delta, dim=0)
    lam = u @ C @ u
    rank = int((ev > lam).sum())
    share = tail * (1 - tail) * delta.square().sum() / lam
    pr = 1 / (u ** 4).sum()
    print(f"    opener {j}: |Δ| {delta.norm():.3g} = {delta.norm() / ev.median().sqrt():.1f} × typical PC spread | "
          f"variance along Δ ranks as PC {rank} | cluster share of it {share:.2f} | coordinate PR {pr:.0f}")
print(f"    reference: coordinate PR of a random direction ≈ {D / 3:.0f}")
if cli.compare:
    p2, res2 = load_run(cli.compare)
    reps2, _ = openers(res2)
    coord = lambda r, idx: ((held.z() - r["MU"].to(dev)) @ (r["IROOT"].to(dev) @ r["O"][:, idx].to(dev))).reshape(-1, len(idx))
    a, b = coord(res, reps), coord(res2, reps2)
    a, b = (a - a.mean(0)) / a.std(0), (b - b.mean(0)) / b.std(0)
    corr = (a.T @ b / (len(a) - 1)).abs()
    print(f"  best |correlation| with an opener of {p2.name}: {fmt(corr.max(1).values)}")
