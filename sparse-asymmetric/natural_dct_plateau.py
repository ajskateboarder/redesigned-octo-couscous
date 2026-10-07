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
from transformers import GPT2Tokenizer

from bilinear_attention_pr_experiment import SlicedModel
from intermediate_mlp_pr_experiment import collect_middle_inputs, read_texts
from scripts.lib.superellipse import robust_p_fit
from tensor_model import load_tensor_gpt

length, targets, extent, size, anchor_count = 12, [4, 8, 11], 16.0, 31, 16
out = ROOT / "natural_dct" / f"L{length}_t{'-'.join(map(str, targets))}"
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
span_l, span_r = (F.normalize(torch.linalg.qr(x).Q @ torch.randn(16, 16, generator=generator).to(device), dim=0).T for x in (L, R))
families = {"dct": (dct_l, dct_r), "span": (span_l, span_r)}

ids = tokenizer(held, return_tensors="pt", truncation=True, max_length=length).input_ids.to(device)
with torch.no_grad():
    start = model.state_before_block(ids, 8)
    block = model.transformer.h[8]
    values = block.lambdas[0] * start["values"] + block.lambdas[1] * start["initial_values"]
    attention, _ = block.attn(F.rms_norm(values, (1152,)), start["first_values"])
    inputs = F.rms_norm(values + attention, (1152,))[:, 1:].reshape(-1, 1152)
sigma = (inputs @ F.normalize(torch.randn(1152, 256, generator=torch.Generator().manual_seed(37)).to(device), dim=0)).std(0).median().item()

coords = np.linspace(-1.0, 1.0, size)
center = size // 2
grid = torch.tensor(coords, dtype=torch.float32, device=device)
a_unit, b_unit = (x.flatten() for x in torch.meshgrid(grid, grid, indexing="ij"))
quadrants = [(slice(center, None), slice(center, None)), (slice(center, None, -1), slice(center, None)),
             (slice(center, None, -1), slice(center, None, -1)), (slice(center, None), slice(center, None, -1))]
axis = np.linspace(0.0, extent, center + 1)


def p_of(grids):
    ps = []
    for i, j in quadrants:
        fits = robust_p_fit(axis, grids[:, i, j], thresholds=("50", "75"), max_alphas=(extent / 2, 3 * extent / 4, extent),
                            n_bootstrap=0, exact_geodesic=False)
        ps.append(np.nanmedian([c["p_point"] for k, c in fits.items() if k != "__levels__"]))
    return np.nanmedian(ps)


def run(op, values, theta_l, theta_r, linear=(), pre0=None):
    # SlicedModel.forward with the swish of chosen blocks replaced by its tangent at the clean pre-activation
    n = len(values)
    initial = op.initial_values.expand(n, -1, -1)
    first = op.first_values.expand(n, *op.first_values.shape[1:]).contiguous()
    pres = []
    for rel, block in enumerate(op.blocks):
        values = block.lambdas[0] * values + block.lambdas[1] * initial
        attention, first, _ = op.attn(block.attn, F.rms_norm(values, (1152,)), first)
        values = values + attention
        x = F.rms_norm(values, (1152,))
        left = block.mlp.left(x + (theta_l if rel == 0 else 0.0))
        right = block.mlp.right(x + (theta_r if rel == 0 else 0.0))
        pres.append(left)
        if rel in linear:
            s = torch.sigmoid(pre0[rel])
            gate = pre0[rel] * s + s * (1 + pre0[rel] * (1 - s)) * (left - pre0[rel])
        else:
            gate = left * torch.sigmoid(left)
        values = values + block.mlp.Down(gate * right) + block.mlp.Down_bias
    return values[:, targets].mean(1), pres


state8 = collect_middle_inputs(model, tokenizer, anchors, 8, length)
ops8 = [SlicedModel(model, tuple(s[c:c + 1] for s in state8), 8, 12, targets) for c in range(len(anchors))]
with torch.no_grad():
    clean_pre = [run(op, op.reference_values, 0.0, 0.0)[1] for op in ops8]
    probe = 3 * sigma * dct_l[0][None, None]
    mine = run(ops8[0], ops8[0].reference_values, probe, probe)[0]
    theirs = ops8[0](probe, probe, ops8[0].reference_values, torch.zeros(1, 1152, device=device))[0]
    print(f"forward check vs SlicedModel: max |diff| {(mine - theirs).abs().max().item():.2e} (target norm {theirs.norm().item():.0f})")


@torch.no_grad()
def source_plane(l, r, linear, chunk=2048):
    # (a): gate = a·l, up = b·r on the layer-8 MLP input, all positions
    theta_l, theta_r = (a_unit * extent * sigma)[:, None] * l, (b_unit * extent * sigma)[:, None] * r
    grids = []
    for op, pre0 in zip(ops8, clean_pre):
        y = torch.cat([run(op, op.reference_values.expand(len(tl), -1, -1), tl[:, None], tr[:, None], linear, pre0)[0]
                       for tl, tr in zip(theta_l.split(chunk), theta_r.split(chunk))])
        grids.append((y - y[len(y) // 2]).norm(dim=-1))
    return torch.stack(grids).view(-1, size, size).cpu().numpy()


@torch.no_grad()
def residual_plane(ops, sd, l, r, chunk=2048):
    # (b): residual push a·l + b·r at the input of an earlier block, each axis in its own data SD there
    theta = (a_unit * extent * sd[0])[:, None] * l + (b_unit * extent * sd[1])[:, None] * r
    zero = torch.zeros(1152, device=device)
    grids = []
    for op in ops:
        y = torch.cat([op(zero, zero, op.reference_values + t[:, None], torch.zeros(len(t), 1152, device=device))[0] for t in theta.split(chunk)])
        grids.append((y - y[len(y) // 2]).norm(dim=-1))
    return torch.stack(grids).view(-1, size, size).cpu().numpy()


results = {}
for name, linear in (("native", ()), ("swish linear @8", (0,)), ("swish linear @8-11", (0, 1, 2, 3))):
    for family, (ls, rs) in families.items():
        results["a", name, family] = np.array([p_of(source_plane(l, r, linear)) for l, r in zip(ls, rs)])
    print(f"done (a) {name}", flush=True)

for k in (8, 6, 4):
    state = collect_middle_inputs(model, tokenizer, anchors, k, length)
    ops = [SlicedModel(model, tuple(s[c:c + 1] for s in state), k, 12, targets) for c in range(len(anchors))]
    with torch.no_grad():
        residual = model.state_before_block(ids, k)["values"][:, 1:].reshape(-1, 1152).float()
    for family, (ls, rs) in families.items():
        results["b", f"residual @{k}", family] = np.array([
            p_of(residual_plane(ops, torch.stack([(residual @ l).std(), (residual @ r).std()]), l, r)) for l, r in zip(ls, rs)])
    del state, ops
    print(f"done (b) residual @{k}", flush=True)
np.savez(out / "plateau_p.npz", **{" | ".join(key): v for key, v in results.items()})

print(f"\nsuperellipse p of the (l_i, r_i) plane, ±{extent:g}, layer-12 target {targets}, {anchor_count} anchors; median [IQR] over 16 factors")
for part, reference in (("a", "native"), ("b", "residual @8")):
    print(f"\n({part}) " + ("source push, gate=a·l / up=b·r on layer-8 MLP input, σ units" if part == "a"
                         else "residual push a·l + b·r at block input, per-axis data SD units"))
    for (p_, arm, family), v in results.items():
        if p_ != part:
            continue
        delta = v - results[part, reference, family]
        print(f"  {arm:20s} {family:5s} | p {np.nanmedian(v):.2f} [{np.nanpercentile(v, 25):.2f}-{np.nanpercentile(v, 75):.2f}] "
              f"p<1 {np.sum(v < 1):2d}/16 | Δ vs {reference}: {np.nanmedian(delta):+.2f}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, part in zip(axes, ("a", "b")):
    arms = [arm for p_, arm, family in results if p_ == part and family == "dct"]
    for offset, family, color in ((-0.12, "dct", "C3"), (0.12, "span", "C0")):
        for x, arm in enumerate(arms):
            v = results[part, arm, family]
            ax.scatter(np.full(16, x + offset), v, s=10, c=color, alpha=0.6, label=family if x == 0 else None)
            ax.hlines(np.nanmedian(v), x + offset - 0.1, x + offset + 0.1, color=color)
    ax.axhline(1, color="k", lw=0.5, ls="--")
    ax.set_xticks(range(len(arms)), arms, fontsize=8)
    ax.set_title("(a) swish linearised" if part == "a" else "(b) push from earlier blocks", fontsize=9)
axes[0].set_ylabel("superellipse p")
axes[0].legend(fontsize=8)
fig.tight_layout()
fig.savefig(out / "plateau_p.png", dpi=120)
