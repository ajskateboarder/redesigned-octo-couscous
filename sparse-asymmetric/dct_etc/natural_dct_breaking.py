import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for path in (str(ROOT.parent), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.nn import functional as F
from transformers import GPT2Tokenizer

from bilinear_attention_pr_experiment import SlicedModel
from intermediate_mlp_pr_experiment import collect_middle_inputs, read_texts
from tensor_model import load_tensor_gpt

length, targets, anchor_count = 12, [4, 8, 11], 16
out = ROOT / "natural_dct" / f"L{length}_t{'-'.join(map(str, targets))}"
device = torch.device("cuda")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model, _, _ = load_tensor_gpt("Elriggs/gpt2-swiglu-18l-9h-1152embd-v2", device)
model.requires_grad_(False)
pool = [t for t in read_texts(0, 520)[1] if len(tokenizer(t).input_ids) >= length]
anchors = pool[32:32 + anchor_count]

L, R = torch.load(out / "dct_seed0.pt")
dct_l, dct_r = (F.normalize(x, dim=0).T for x in (L, R))
generator = torch.Generator().manual_seed(31)
span_l, span_r = (F.normalize(torch.linalg.qr(x).Q @ torch.randn(16, 16, generator=generator).to(device), dim=0).T for x in (L, R))
families = {"dct": (dct_l, dct_r), "span": (span_l, span_r)}
randoms = F.normalize(torch.randn(8, 1152, generator=torch.Generator().manual_seed(7)).to(device), dim=-1)

state = collect_middle_inputs(model, tokenizer, anchors, 8, length)
readouts = {12: [SlicedModel(model, tuple(s[c:c + 1] for s in state), 8, 12, targets) for c in range(anchor_count)],
            17: [SlicedModel(model, tuple(s[c:c + 1] for s in state), 8, 17, targets) for c in range(anchor_count)]}
angles = torch.deg2rad(torch.linspace(0, 60, 61, device=device))
signs = torch.tensor([[1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=torch.float32, device=device)


@torch.no_grad()
def curves(u, v, readout):
    # FSEC norm-matched push of the block-8 residual at all positions: rotate each position's activation toward the direction by angle α
    zero = torch.zeros(1152, device=device)
    responses = []
    for op in readouts[readout]:
        a = op.reference_values[0]
        a_hat = F.normalize(a, dim=-1)
        u_t = F.normalize(u - (a_hat @ u)[:, None] * a_hat, dim=-1)
        v_t = v - (a_hat @ v)[:, None] * a_hat
        v_t = F.normalize(v_t - (v_t * u_t).sum(-1, keepdim=True) * u_t, dim=-1)
        r_t = F.normalize(randoms[:, None] - (randoms @ a_hat.T)[..., None] * a_hat, dim=-1)
        directions = torch.cat([torch.stack([u_t, -u_t, v_t, -v_t]),
                                (signs[:, 0, None, None] * u_t + signs[:, 1, None, None] * v_t) / 2 ** 0.5, r_t])
        perturbed = (torch.cos(angles)[:, None, None, None] * a + torch.sin(angles)[:, None, None, None] * a.norm(dim=-1, keepdim=True) * directions)
        y = torch.cat([op(zero, zero, x, torch.zeros(len(x), 1152, device=device))[0] for x in perturbed.reshape(-1, length, 1152).split(2048)])
        responses.append((y - y[0]).norm(dim=-1).view(len(angles), len(directions)))
    response = torch.stack(responses).median(0).values.T.cpu().numpy()
    return {"l": response[:2], "r": response[2:4], "mix": response[4:8], "random": np.median(response[8:], 0)}


def breaking(curve, threshold):
    crossed = np.nonzero(curve > threshold)[0]
    return np.rad2deg(angles[crossed[0]].item()) if len(crossed) else np.nan


cache = out / "plateau_breaking_raw.npz"
if cache.exists():
    raw = dict(np.load(cache))
else:
    raw = {}
    for readout in readouts:
        for family, (ls, rs) in families.items():
            for i, (l, r) in enumerate(zip(ls, rs)):
                for name, value in curves(l, r, readout).items():
                    raw[f"{readout}|{family}|{i}|{name}"] = value
        print(f"done readout layer {readout}", flush=True)
    np.savez(cache, **raw)


def quadrant(readout, family, i, q):
    # the mixture s1·l + s2·r is compared against s1·l and s2·r alone, with the threshold set within that quadrant
    s1, s2 = signs[q].tolist()
    get = lambda name: raw[f"{readout}|{family}|{i}|{name}"]
    picked = {"l": get("l")[0 if s1 > 0 else 1], "r": get("r")[0 if s2 > 0 else 1], "mix": get("mix")[q], "random": get("random")}
    threshold = 0.5 * min(picked["l"].max(), picked["r"].max())
    angle = {name: breaking(picked[name], threshold) for name in ("l", "r", "mix")}
    return {"curves": picked, "threshold": threshold, "angle": angle, "gap": np.nanmin([angle["l"], angle["r"]]) - angle["mix"]}


results = {(readout, family, i): [quadrant(readout, family, i, q) for q in range(4)]
           for readout in readouts for family in families for i in range(16)}


def factor_median(quads):
    # the quadrant whose gap is closest to this factor's median gap over the four sign combinations
    gaps = np.array([q["gap"] for q in quads])
    if not np.isfinite(gaps).any():
        return quads[0]
    return quads[int(np.nanargmin(np.abs(gaps - np.nanmedian(gaps))))]


print("plateau-breaking angles (degrees), per sign quadrant: threshold = 0.5 × the smaller single-axis maximum; "
      "medians over 16 factors × 4 quadrants")
for readout in readouts:
    for family in families:
        a = np.array([[q["angle"][n] for n in ("l", "r", "mix")] for i in range(16) for q in results[readout, family, i]])
        earlier = np.sum(a[:, 2] < np.nanmin(a[:, :2], 1))
        print(f"  readout layer {readout} {family:4s}: l {np.nanmedian(a[:, 0]):4.1f}  r {np.nanmedian(a[:, 1]):4.1f}  mix {np.nanmedian(a[:, 2]):4.1f} | "
              f"mix breaks before both pure directions in {earlier}/64 | median gap {np.nanmedian(np.nanmin(a[:, :2], 1) - a[:, 2]):+.1f}")

x = np.rad2deg(angles.cpu().numpy())
fig, axes = plt.subplots(2, 8, figsize=(22, 6), squeeze=False)
for row, readout in zip(axes, readouts):
    for i, ax in enumerate(row):
        res = factor_median(results[readout, "dct", i])
        for name, color in (("l", "C0"), ("r", "C1"), ("mix", "C2")):
            ax.plot(x, res["curves"][name], color=color, label=name if i == 0 else None)
            if np.isfinite(res["angle"][name]):
                ax.axvline(res["angle"][name], color=color, ls=":", lw=0.8)
        ax.plot(x, res["curves"]["random"], color="grey", ls="--", label="random" if i == 0 else None)
        ax.axhline(res["threshold"], color="k", lw=0.5)
        ax.set_title(f"factor {i}, readout layer {readout}", fontsize=8)
        ax.tick_params(labelsize=6)
    row[0].set_ylabel("L2 response (median over anchors)", fontsize=7)
axes[0, 0].legend(fontsize=6)
fig.supxlabel("rotation angle α toward the direction (degrees)", fontsize=8)
fig.tight_layout()
fig.savefig(out / "plateau_breaking_curves.png", dpi=120)

fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
for ax, readout in zip(axes, readouts):
    for family, color in (("dct", "C3"), ("span", "C0")):
        # per factor: median over the four sign quadrants
        a = np.array([np.nanmedian([[np.nanmin([q["angle"]["l"], q["angle"]["r"]]), q["angle"]["mix"]]
                                    for q in results[readout, family, i]], 0) for i in range(16)])
        ax.scatter(a[:, 0], a[:, 1], color=color, label=family, s=18)
    limit = ax.get_xlim()[1]
    ax.plot([0, limit], [0, limit], color="k", lw=0.6)
    ax.set_xlabel("breaking angle of the more sensitive pure direction (deg)")
    ax.set_ylabel("breaking angle of the l/r mixture (deg)")
    ax.set_title(f"readout layer {readout}: below the line = mixture breaks first", fontsize=9)
    ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(out / "plateau_breaking_summary.png", dpi=120)

medians = np.array([np.nanmedian([q["gap"] for q in results[12, "dct", i]]) for i in range(16)])
example = int(np.nanargmin(np.abs(medians - np.nanmedian(medians))))
res = factor_median(results[12, "dct", example])
print(f"example (median factor by gap): dct factor {example}, breaking angles {res['angle']}, gap {res['gap']:+.1f}; "
      f"per-factor median gaps {np.round(medians, 1).tolist()}")
fig, ax = plt.subplots(figsize=(8, 6))
for name, color, label in (("l", "C0", "l alone"), ("r", "C1", "r alone"), ("mix", "C2", "l + r mixture"), ("random", "grey", "random directions")):
    ax.plot(x, res["curves"][name], color=color, lw=2 if name == "mix" else 1.4, ls="--" if name == "random" else "-", label=label)
    if name != "random" and np.isfinite(res["angle"][name]):
        ax.axvline(res["angle"][name], color=color, ls=":", lw=1)
ax.axhline(res["threshold"], color="k", lw=0.8, label=f"plateau-breaking threshold (T = {res['threshold']:.0f})")
random_angle = breaking(res["curves"]["random"], res["threshold"])
cutoff = random_angle + 2 if np.isfinite(random_angle) else x[-1]
print(f"random directions cross the threshold at {random_angle} degrees; plotting up to {cutoff}")
visible = x <= cutoff
ax.set_xlim(0, cutoff)
ax.set_ylim(0, 1.05 * max(res["curves"][n][visible].max() for n in ("l", "r", "mix", "random")))
ax.set_xlabel("rotation angle toward the direction (degrees)")
ax.set_ylabel("layer-12 response (L2, median over anchors)")
ax.set_axisbelow(True)
ax.grid(True, linestyle='-', alpha=0.2, color='grey')
ax.legend(fontsize=9, loc="upper left")
fig.tight_layout()
fig.savefig(out / "plateau_breaking_example.png", dpi=300)
