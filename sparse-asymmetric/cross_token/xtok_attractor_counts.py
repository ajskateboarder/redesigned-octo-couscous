"""Attractor counts for the cross-token DCT, with the protocol of bigram_attractor_counts.py.

64 independent single-factor power iterations (no orthogonalisation) on N = 4, 8, 16, 32 contexts from the model's bigram
chain (the same draws as bigram_attractor_counts.py); an attractor is a group of end points with |cos l|·|cos r| > 0.8
reached by at least 2 runs. The cross-token DCT pushes the residual input of block --S (Σ^½-whitened, Σ from 1000 bigram
windows) and uses the fisher readout of xtok.py. Adds a fourth column to natural_dct/bigram/attractor_counts.png.
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.sparse.csgraph import connected_components
from torch.nn import functional as F

import xtok

parser = argparse.ArgumentParser()
parser.add_argument("--S", type=int, default=12)
parser.add_argument("--iters", type=int, default=25)
args = parser.parse_args()

runs, sizes, default = 64, [4, 8, 16, 32], 0.8
out = xtok.ROOT / "natural_dct" / "bigram"
device = xtok.device
torch.set_float32_matmul_precision("high")


def basins(matrix, threshold):
    _, label = connected_components((matrix > threshold).cpu().numpy(), directed=False)
    return sorted(np.bincount(label), reverse=True)


def attractors(matrix, threshold=default):
    return sum(s >= 2 for s in basins(matrix, threshold))


P = xtok.push_metric(xtok.states(xtok.pool_ids("bigram", 1000), args.S)[0])[0]
results, fits = {}, {}
for n in sizes:
    path = out / f"attractors_crosstoken_S{args.S}_n{n}.pt"
    if not path.exists():
        # seed 1000 + 10n reproduces the first 12 tokens of bigram_attractor_counts.sample(n)
        rows = xtok.Rows(xtok.bigram_ids(n, 1000 + 10 * n), args.S, "fisher", xtok.n_layer)
        fit = xtok.Fitter(rows, P, runs, batch=32, chunk=8).fit(1, "census", args.iters, log=None)
        L, R = (F.normalize(fit[k].T.to(device) @ P, dim=1) for k in "LR")
        torch.save({"L": L.cpu(), "R": R.cpu(), "E": fit["E"], "iterations": fit["iterations"], "seconds": fit["seconds"]}, path)
    saved = torch.load(path)
    L, R = fits[n] = saved["L"], saved["R"]
    results[n] = (L @ L.T).abs() * (R @ R.T).abs()
    print(f"N = {n:2d}: {attractors(results[n])} attractors reached by ≥ 2 runs ({saved['iterations']} sweeps, "
          f"{saved['seconds']:.0f} s); basin sizes {basins(results[n], default)[:10]}; attractors at threshold 0.6 / 0.95: "
          f"{attractors(results[n], 0.6)} / {attractors(results[n], 0.95)}", flush=True)

top = {n: int((results[n] > default).sum(1).argmax()) for n in sizes}
print("\nlargest basin at each data size, |cos l|·|cos r| between their representatives:")
for a in sizes:
    row = [((fits[a][0][top[a]] @ fits[b][0][top[b]]).abs() * (fits[a][1][top[a]] @ fits[b][1][top[b]]).abs()).item() for b in sizes]
    print(f"  N = {a:2d}: " + " ".join(f"{v:.2f}" for v in row))

# the post's figure with the cross-token DCT as a fourth column
stems = {"Amp": "amp_whitened", "quadratic DCT": "quadratic", "asymmetric DCT": "asymmetric"}
forms = {"Amp": "v → v", "quadratic DCT": "v·v → u", "asymmetric DCT": "l·r → u", "cross-token DCT": "l (earlier) · r (current)"}


def similarity(method, parts):
    if method == "Amp":
        return (parts[0].T @ parts[0]).abs()
    u, l = parts[0], parts[1]
    r = parts[2] if method == "asymmetric DCT" else l
    return (u.T @ u).abs() * (l.T @ l).abs() * (r.T @ r).abs()


counts = {m: [attractors(similarity(m, torch.load(out / f"attractors_{s}_n{n}.pt"))) for n in sizes] for m, s in stems.items()}
counts["cross-token DCT"] = [attractors(results[n]) for n in sizes]
for method, c in counts.items():
    print(f"{method:15s}: {c}")

slate, muted, coral = "#4b5d6e", "#6b7c8e", "#ee7853"
plt.rcParams["font.family"] = "serif"
offsets = np.linspace(-0.24, 0.24, len(sizes))
fig, ax = plt.subplots(figsize=(8.6, 4.8))
for k, (method, c) in enumerate(counts.items()):
    median = ax.hlines(np.median(c), k - 0.33, k + 0.33, color=coral, lw=3.5, zorder=2)
    dots = ax.scatter(k + offsets, c, s=42, color=slate, alpha=0.75, edgecolors="none", zorder=3)
    for x, y, n in zip(k + offsets, c, sizes):
        ax.annotate(str(n), (x, y), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=8, color=muted)
ax.set_xticks(range(len(counts)), [f"{m}\n({forms[m]})" for m in counts])
ax.set_xlim(-0.6, len(counts) - 0.4)
ax.set_ylim(-0.6, 13.6)
ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(2))
ax.tick_params(colors=muted, labelsize=11)
ax.spines[["top", "right"]].set_visible(False)
ax.spines[["left", "bottom"]].set_color(muted)
ax.set_ylabel(f"Attractors ({runs} random starts)", fontsize=14, color="#33475b")
ax.set_axisbelow(True)
ax.grid(True, axis="y", color="#e5e5e5", lw=0.8)
ax.legend([dots, median], ["fits (label: bigram contexts)", "median over data sizes"], loc="upper left", fontsize=11)
fig.tight_layout()
fig.savefig(out / "attractor_counts_crosstoken.png", dpi=300)
print(f"saved {out / 'attractor_counts_crosstoken.png'}")
