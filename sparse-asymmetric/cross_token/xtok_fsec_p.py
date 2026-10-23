"""FSEC superellipse exponents for cross-token DCT factors, following natural_dct_fsec_p.py.

FSEC's norm-preserving push, mapped onto the cross-token factor: in the (+,+) quadrant, every earlier position 1..10 of a
12-token anchor window is rotated by angle α toward l (each in its own plane spanned by the activation and l) and the last
position 11 by angle β toward r, at the residual input of block S. The response is the L2 change of the final residual
stream at position 11 (FSEC reads a later layer at the pushed position; the attractors gate as late as MLP 17), median
over 16 held-out WikiText anchors, fitted with FSEC's unchanged superellipse code (FranciscoHS/fsec-paper, /tmp/fsec).

Pair types, each against nulls:
    l, r   cross-position (l at earlier positions, r at the current one); nulls: random mixtures of the same spans,
           mispaired factors (l_i, r_i+1); also the (l, -r) quadrant since the factor signs are arbitrary
    l, l   both rotated together at the earlier positions (FSEC's single-site geodesic per position)
    r, r   both at the current position, exactly FSEC's protocol; PCA pairs (FSEC's baseline) likewise
For cross-position pairs the two angles live on separate spheres, so contour points are normalised per axis
(sin α / sin α*, sin β / sin β*) instead of FSEC's single-sphere geodesic. Also reports the cosine between the two
single-axis response vectors (p = 1 without any interaction if they are aligned, p = 2 if orthogonal) and FSEC-style
plateau-breaking angles along the axes and the diagonal α = β.

Usage: python xtok_fsec_p.py [--fit cross_token/S12_fisher_model_joint_w16_st450_seed101.pt]
"""
import argparse
import itertools
import sys

sys.path.insert(0, "/tmp/fsec")

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.nn import functional as F

from scripts.lib.superellipse import axis_intercept, extract_contour, fit_superellipse, threshold_levels

import xtok

parser = argparse.ArgumentParser()
parser.add_argument("--fit", default="cross_token/S12_fisher_model_joint_w16_st450_seed101.pt")
parser.add_argument("--anchors", type=int, default=16)
parser.add_argument("--readout", choices=["residual", "written", "kl"], default="residual",
                    help="residual: L2 change of the final residual (FSEC); written: the same minus the pushed position's own "
                         "change carried by the skip connection; kl: KL(clean || pushed) of the next-token distribution")
args = parser.parse_args()

device, D, t = xtok.device, xtok.D, xtok.length - 1
steps, max_deg = 31, 60.0
angles = np.linspace(0.0, max_deg, steps)
fit = torch.load(xtok.ROOT / args.fit, weights_only=False)
S = fit["args"]["S"]
stem = (xtok.ROOT / args.fit).stem
cache = xtok.out / f"{stem}_fsec_p{'' if args.readout == 'residual' else '_' + args.readout}.npz"
groups = ["lr", "ll", "rr", "pca"]


def fit_p(grids, separate):
    # FSEC's robust protocol: thresholds 50/75% of the smaller axis maximum, windows 30/45/60°
    median = np.median(grids, axis=0)
    levels = threshold_levels(median)
    ps, errors = [], []
    for label in ("50", "75"):
        threshold = levels[label]
        t1 = axis_intercept(angles, median[:, 0], threshold)
        t2 = axis_intercept(angles, median[0, :], threshold)
        for window in (max_deg / 2, 3 * max_deg / 4, max_deg):
            raw = extract_contour(angles, median, threshold)
            if len(raw):
                raw = raw[(raw[:, 0] < angles.max() - 1) & (raw[:, 1] < angles.max() - 1)]
                raw = raw[(raw[:, 0] <= window) & (raw[:, 1] <= window)]
            if len(raw) < 3 or not (np.isfinite(t1) and np.isfinite(t2) and t1 > 0 and t2 > 0):
                ps.append(np.nan)
                errors.append(np.nan)
                continue
            a1, a2 = np.deg2rad(raw[:, 0]), np.deg2rad(raw[:, 1])
            if separate:
                x, y = np.sin(a1), np.sin(a2)
            else:
                rho = np.sqrt(np.maximum(a1 ** 2 + a2 ** 2, 1e-24))
                x, y = np.sin(rho) / rho * a1, np.sin(rho) / rho * a2
            result = fit_superellipse(x / np.sin(np.deg2rad(t1)), y / np.sin(np.deg2rad(t2)))
            ps.append(result["p"])
            errors.append(result["mean_radial_frac"])
    return np.nanmedian(ps), np.nanmedian(errors)


def breaking(median):
    # first angle where the response exceeds half the smaller axis maximum: along α, along β, and along α = β
    # (total rotation √2·α, the same budget as FSEC's 45° mixture)
    threshold = 0.5 * min(median[:, 0].max(), median[0, :].max())
    first = lambda curve, scale: scale * angles[np.nonzero(curve > threshold)[0][0]] if (curve > threshold).any() else np.nan
    return first(median[:, 0], 1.0), first(median[0, :], 1.0), first(np.diag(median), 2 ** 0.5)


def compute():
    torch.set_float32_matmul_precision("high")
    P = fit["P"].to(device)
    dct_l, dct_r = (F.normalize(fit[k].T.to(device) @ P, dim=1) for k in "LR")
    generator = torch.Generator().manual_seed(31)
    span_l, span_r = (F.normalize(torch.randn(16, 16, generator=generator).to(device) @ torch.linalg.qr(x.T).Q.T, dim=1)
                      for x in (dct_l, dct_r))
    m, initial, first = xtok.states(xtok.wikitext_ids(0, args.anchors), S)
    held = xtok.states(xtok.wikitext_ids(16, 256), S)[0][:, 1:].reshape(-1, D)
    pcs = torch.linalg.svd(held - held.mean(0), full_matrices=False).Vh[:16]
    with torch.no_grad():
        clean = xtok.run(m, initial, first, S, xtok.n_layer)[:, t]
        clean_lp = torch.log_softmax(xtok.logits(clean), -1)
        # the skip connection carries the input of block S to the final residual scaled by the later blocks' λ0
        skip = torch.stack([xtok.blocks[b].lambdas[0] for b in range(S + 1, xtok.n_layer)]).prod()
    chart = torch.deg2rad(torch.tensor(angles, dtype=torch.float32, device=device))
    A1, A2 = (x.flatten() for x in torch.meshgrid(chart, chart, indexing="ij"))
    earlier = torch.arange(1, t, device=device)

    def rotate(x, d1, d2, a1, a2):
        # exp-map chart at each activation in x (..., D): moved = |x| (cos ρ x̂ + sin ρ v/ρ), v = a1 p1 + a2 p2
        norm = x.norm(dim=-1, keepdim=True)
        x_hat = x / norm
        p1 = F.normalize(d1 - (x_hat @ d1)[..., None] * x_hat, dim=-1)
        if d2 is None:
            v = a1[:, None, None] * p1
        else:
            p2 = d2 - (x_hat @ d2)[..., None] * x_hat
            p2 = F.normalize(p2 - (p2 * p1).sum(-1, keepdim=True) * p1, dim=-1)
            v = a1[:, None, None] * p1 + a2[:, None, None] * p2
        rho = v.norm(dim=-1, keepdim=True)
        coeff = torch.where(rho > 1e-12, torch.sin(rho) / rho.clamp_min(1e-12), torch.ones_like(rho))
        return norm * (torch.cos(rho) * x_hat + coeff * v)

    @torch.no_grad()
    def grids(d1, d2, site):
        out_grids, cosines = [], []
        for c in range(args.anchors):
            x = m[c].expand(len(A1), -1, -1).clone()
            if site == "cross":
                x[:, earlier] = rotate(m[c, earlier], d1, None, A1, None)
                x[:, t] = rotate(m[c, t:t + 1], d2, None, A2, None)[:, 0]
            elif site == "earlier":
                x[:, earlier] = rotate(m[c, earlier], d1, d2, A1, A2)
            else:
                x[:, t] = rotate(m[c, t:t + 1], d1, d2, A1, A2)[:, 0]
            final = torch.cat([xtok.run(z, initial[c:c + 1].expand(len(z), -1, -1), first[c:c + 1].expand(len(z), *first.shape[1:]),
                                        S, xtok.n_layer)[:, t] for z in x.split(1024)])
            y = final - clean[c]
            if args.readout == "written":
                y = y - skip * (x[:, t] - m[c, t])
            if args.readout == "kl":
                lp = torch.log_softmax(xtok.logits(final), -1)
                grid = (clean_lp[c].exp() * (clean_lp[c] - lp)).sum(-1).view(steps, steps)
            else:
                grid = y.norm(dim=-1).view(steps, steps)
            out_grids.append(grid)
            cosines.append(F.cosine_similarity(y.view(steps, steps, D)[-1, 0], y.view(steps, steps, D)[0, -1], dim=0).item())
        return torch.stack(out_grids).cpu().numpy(), float(np.median(cosines))

    pairs16 = list(itertools.combinations(range(16), 2))
    pairs = {
        ("lr", "dct"): ([(dct_l[i], dct_r[i]) for i in range(16)], "cross"),
        ("lr", "dct, -r"): ([(dct_l[i], -dct_r[i]) for i in range(16)], "cross"),
        ("lr", "mispaired"): ([(dct_l[i], dct_r[(i + 1) % 16]) for i in range(16)], "cross"),
        ("lr", "baseline"): ([(span_l[i], span_r[i]) for i in range(16)], "cross"),
        ("ll", "dct"): ([(dct_l[i], dct_l[j]) for i, j in pairs16], "earlier"),
        ("ll", "baseline"): ([(span_l[i], span_l[j]) for i, j in pairs16], "earlier"),
        ("rr", "dct"): ([(dct_r[i], dct_r[j]) for i, j in pairs16], "current"),
        ("rr", "baseline"): ([(span_r[i], span_r[j]) for i, j in pairs16], "current"),
        ("pca", "baseline"): ([(pcs[i], pcs[j]) for i, j in pairs16], "current"),
    }
    results = {}
    for (group, kind), (items, site) in pairs.items():
        rows = []
        for d1, d2 in items:
            g, cosine = grids(d1, d2, site)
            p, error = fit_p(g, site == "cross")
            rows.append((p, error, cosine, *breaking(np.median(g, axis=0))))
        results[f"{group}_{kind}"] = np.array(rows)
        print(f"done {group} {kind}", flush=True)
    np.savez(cache, **results)
    return results


results = dict(np.load(cache)) if cache.exists() else compute()

print(f"\n{stem}: FSEC superellipse p, (+,+) quadrant, 0-{max_deg:g}°, {args.readout} readout at the last position, "
      f"{args.anchors} anchors; good fit = mean radial error < 0.1")
for key, a in results.items():
    good = a[:, 1] < 0.1
    p = a[good, 0]
    print(f"  {key:16s} good fits {good.sum():3d}/{len(a)} | median p {np.median(p):.2f} [IQR {np.percentile(p, 25):.2f}-"
          f"{np.percentile(p, 75):.2f}] | p < 1: {np.mean(p < 1):.0%}, p > 2: {np.mean(p > 2):.0%} | median cos(single-axis "
          f"responses) {np.median(a[:, 2]):+.2f}")
print("\nplateau breaking (first angle above half the smaller axis maximum), medians: α only / β only / α = β (total √2·α)")
for key in ("lr_dct", "lr_dct, -r", "lr_mispaired", "lr_baseline"):
    a = results[key]
    first_axis = np.nanmin(a[:, 3:5], axis=1)
    print(f"  {key:16s} {np.nanmedian(a[:, 3]):.1f}° / {np.nanmedian(a[:, 4]):.1f}° / {np.nanmedian(a[:, 5]):.1f}°; diagonal "
          f"breaks before the earlier axis in {np.mean(a[:, 5] < first_axis):.0%} of factors")
p0 = results["lr_dct"][0]
print(f"\ntop factor (largest objective): p {p0[0]:.2f} (fit error {p0[1]:.3f}), cos {p0[2]:+.2f}; with -r: "
      f"p {results['lr_dct, -r'][0][0]:.2f} (fit error {results['lr_dct, -r'][0][1]:.3f})")

slate, muted, coral = "#4b5d6e", "#6b7c8e", "#ee7853"
plt.rcParams["font.family"] = "serif"
labels = {"lr": "l (earlier), r (current)\n(same factor)", "ll": "l, l\n(different factors)", "rr": "r, r\n(different factors)",
          "pca": "PCA\n(top 16 components)"}
rng = np.random.default_rng(0)
fig, ax = plt.subplots(figsize=(8, 5.2))
ax.axhline(2, color=muted, lw=1.2, ls="--", zorder=1)
for k, group in enumerate(groups):
    for kind, offset, marker in (("dct", -0.17, "o"), ("baseline", 0.17, "x")):
        key = f"{group}_{kind}"
        if key not in results:
            continue
        a = results[key]
        p = a[a[:, 1] < 0.1, 0]
        centre = k if group == "pca" else k + offset
        x = centre + rng.uniform(-0.09, 0.09, len(p))
        if marker == "o":
            ax.scatter(x, p, s=26, color=slate, alpha=0.6, edgecolors="none", zorder=3)
        elif group == "pca":
            ax.scatter(x, p, s=30, color=muted, alpha=0.55, marker="+", linewidths=1.1, zorder=3)
        else:
            ax.scatter(x, p, s=24, color=muted, alpha=0.55, marker="x", linewidths=1.1, zorder=3)
        ax.hlines(np.median(p), centre - 0.15, centre + 0.15, color=coral, lw=3.5, zorder=4)
handles = [ax.scatter([], [], s=26, color=slate, alpha=0.6, edgecolors="none"),
           ax.scatter([], [], s=24, color=muted, alpha=0.55, marker="x", linewidths=1.1),
           ax.scatter([], [], s=30, color=muted, alpha=0.55, marker="+", linewidths=1.1),
           plt.Line2D([], [], color=coral, lw=3.5), plt.Line2D([], [], color=muted, lw=1.2, ls="--")]
ax.legend(handles, ["cross-token DCT pairs", "random mixtures of\nthe same span", "PCA pairs", "median", "p = 2 (isotropic)"],
          loc="upper right", fontsize=10)
ax.set_xticks(range(len(groups)), [labels[g] for g in groups])
ax.set_xlim(-0.6, len(groups) - 0.4)
ax.tick_params(colors=muted, labelsize=11)
ax.spines[["top", "right"]].set_visible(False)
ax.spines[["left", "bottom"]].set_color(muted)
ax.set_ylabel("Superellipse exponent p", fontsize=14, color="#33475b")
ax.set_title(f"Superellipse exponent by pair type (cross-token DCT, block {S})", fontsize=14)
ax.set_axisbelow(True)
ax.grid(True, axis="y", color="#e5e5e5", lw=0.8)
fig.tight_layout()
fig.savefig(cache.with_suffix(".png"), dpi=300)
print(f"saved {cache.with_suffix('.png')}")
