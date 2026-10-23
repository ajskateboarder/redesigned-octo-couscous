import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for path in (str(ROOT.parent), str(ROOT), "/tmp/fsec"):
    if path not in sys.path:
        sys.path.insert(0, path)

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.nn import functional as F

# FSEC's superellipse code, unchanged (FranciscoHS/fsec-paper scripts/lib/superellipse.py, sha256 47d19c99…)
from scripts.lib.superellipse import axis_intercept, extract_contour, fit_superellipse, robust_p_fit, threshold_levels

length, targets, anchor_count, steps, max_deg = 12, [4, 8, 11], 16, 31, 60.0
out = ROOT / "natural_dct" / f"L{length}_t{'-'.join(map(str, targets))}"
cache = out / "fsec_p_by_pair.npz"
angles = np.linspace(0.0, max_deg, steps)
groups = ["lr", "ll", "rr", "pca"]


def fit_p(grids, overlap):
    # FSEC's robust protocol (thresholds 50/75% of the smaller axis maximum, windows 30/45/60°, exact geodesic
    # normalisation). The normalisation is written for two unit tangent directions with overlap c, so a factor's own
    # (l, r) can be measured without orthogonalising r against l; with c = 0 it is FSEC's normalisation exactly.
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
            rho = np.sqrt(np.maximum(a1 ** 2 + a2 ** 2 + 2 * overlap * a1 * a2, 1e-24))
            scale = np.sin(rho) / rho
            fit = fit_superellipse(scale * a1 / np.sin(np.deg2rad(t1)), scale * a2 / np.sin(np.deg2rad(t2)))
            ps.append(fit["p"])
            errors.append(fit["mean_radial_frac"])
    return np.nanmedian(ps), np.nanmedian(errors)


def compute():
    from transformers import GPT2Tokenizer

    from bilinear_attention_pr_experiment import SlicedModel
    from intermediate_mlp_pr_experiment import collect_middle_inputs, read_texts
    from tensor_model import load_tensor_gpt

    device = torch.device("cuda")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model, _, _ = load_tensor_gpt("Elriggs/gpt2-swiglu-18l-9h-1152embd-v2", device)
    model.requires_grad_(False)
    pool = [t for t in read_texts(0, 520)[1] if len(tokenizer(t).input_ids) >= length]
    anchors, held = pool[32:32 + anchor_count], pool[56:]

    L, R = torch.load(out / "dct_seed0.pt")
    dct_l, dct_r = (F.normalize(x, dim=0).T for x in (L, R))
    generator = torch.Generator().manual_seed(31)
    span_l, span_r = (F.normalize(torch.linalg.qr(x).Q @ torch.randn(16, 16, generator=generator).to(device), dim=0).T
                      for x in (L, R))
    ids = tokenizer(held, return_tensors="pt", truncation=True, max_length=length).input_ids.to(device)
    with torch.no_grad():
        residual = model.state_before_block(ids, 8)["values"][:, 1:].reshape(-1, 1152).float()
    pcs = torch.linalg.svd(residual - residual.mean(0), full_matrices=False).Vh[:16]

    # FSEC protocol: rotate only the last position's block-8 residual, read the layer-12 residual at that position
    state = collect_middle_inputs(model, tokenizer, anchors, 8, length)
    ops = [SlicedModel(model, tuple(s[c:c + 1] for s in state), 8, 12, [length - 1]) for c in range(anchor_count)]
    chart = torch.deg2rad(torch.tensor(angles, dtype=torch.float32, device=device))
    A1, A2 = (x.flatten() for x in torch.meshgrid(chart, chart, indexing="ij"))
    zero = torch.zeros(1152, device=device)

    @torch.no_grad()
    def grids(d1, d2, orthogonalize, chunk=2048):
        # exp-map chart on the (+,+) quadrant: a' = |a| (cos ρ â + sin ρ v/ρ), v = α1 p1 + α2 p2, ρ = |v|
        out_grids, overlaps = [], []
        for op in ops:
            x0 = op.reference_values[0]
            last = x0[length - 1]
            norm = last.norm()
            a_hat = last / norm
            p1 = F.normalize(d1 - (a_hat @ d1) * a_hat, dim=0)
            p2 = d2 - (a_hat @ d2) * a_hat
            if orthogonalize:
                p2 = p2 - (p2 @ p1) * p1
            p2 = F.normalize(p2, dim=0)
            v = A1[:, None] * p1 + A2[:, None] * p2
            rho = v.norm(dim=-1)
            coeff = torch.where(rho > 1e-12, torch.sin(rho) / rho.clamp_min(1e-12), torch.ones_like(rho))
            moved = norm * (torch.cos(rho)[:, None] * a_hat + coeff[:, None] * v)
            x = x0.expand(len(rho), -1, -1).clone()
            x[:, length - 1] = moved
            y = torch.cat([op(zero, zero, c, torch.zeros(len(c), 1152, device=device))[0] for c in x.split(chunk)])
            out_grids.append((y - y[0]).norm(dim=-1).view(steps, steps))
            overlaps.append((p1 @ p2).item())
        return torch.stack(out_grids).cpu().numpy(), float(np.median(overlaps))

    everything = list(itertools.combinations(range(16), 2))
    pairs = {
        ("lr", "dct"): ([(dct_l[i], dct_r[i]) for i in range(16)], False),
        ("lr", "baseline"): ([(span_l[i], span_r[i]) for i in range(16)], False),
        ("ll", "dct"): ([(dct_l[i], dct_l[j]) for i, j in everything], True),
        ("ll", "baseline"): ([(span_l[i], span_l[j]) for i, j in everything], True),
        ("rr", "dct"): ([(dct_r[i], dct_r[j]) for i, j in everything], True),
        ("rr", "baseline"): ([(span_r[i], span_r[j]) for i, j in everything], True),
        ("pca", "baseline"): ([(pcs[i], pcs[j]) for i, j in everything], True),
    }
    results = {}
    for (group, kind), (items, orthogonalize) in pairs.items():
        rows = []
        for k, (d1, d2) in enumerate(items):
            g, overlap = grids(d1, d2, orthogonalize)
            p, error = fit_p(g, overlap)
            if orthogonalize and k < 3:
                # check against FSEC's own fitter on the same grid
                reference = robust_p_fit(angles, g, thresholds=("50", "75"), max_alphas=(max_deg / 2, 3 * max_deg / 4, max_deg),
                                         n_bootstrap=0, exact_geodesic=True)
                p_ref = np.nanmedian([v["p_point"] for key, v in reference.items() if key != "__levels__"])
                assert np.isclose(p, p_ref, equal_nan=True), (p, p_ref)
            rows.append((p, error, overlap))
        results[f"{group}_{kind}"] = np.array(rows)
        a = results[f"{group}_{kind}"]
        good = a[:, 1] < 0.1
        print(f"{group:3s} {kind:8s}: good fits {good.sum():3d}/{len(a)} | median p {np.median(a[good, 0]):.2f} | "
              f"median overlap {np.median(a[:, 2]):+.2f}", flush=True)
    np.savez(cache, **results)
    return results


results = dict(np.load(cache)) if cache.exists() else compute()

slate, muted, coral = "#4b5d6e", "#6b7c8e", "#ee7853"
plt.rcParams["font.family"] = "serif"
labels = {"lr": "l, r\n(same factor)", "ll": "l, l\n(different factors)", "rr": "r, r\n(different factors)",
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
ax.legend(handles, ["DCT input pairs", "random mixtures of\nthe same span", "PCA pairs", "median", "p = 2 (isotropic)"],
          loc="upper right", fontsize=10)
ax.set_xticks(range(len(groups)), [labels[g] for g in groups])
ax.set_xlim(-0.6, len(groups) - 0.4)
ax.tick_params(colors=muted, labelsize=11)
ax.spines[["top", "right"]].set_visible(False)
ax.spines[["left", "bottom"]].set_color(muted)
ax.set_ylabel("Superellipse exponent p", fontsize=14, color="#33475b")
ax.set_title("Superellipse exponent by pair type", fontsize=15)
ax.set_axisbelow(True)
ax.grid(True, axis="y", color="#e5e5e5", lw=0.8)
fig.tight_layout()
fig.savefig(out / "fsec_p_by_pair.png", dpi=300)
print(f"saved {out / 'fsec_p_by_pair.png'}")
