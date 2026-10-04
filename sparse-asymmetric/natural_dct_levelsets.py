import argparse
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

parser = argparse.ArgumentParser()
parser.add_argument("--length", type=int, default=12)
parser.add_argument("--targets", type=int, nargs="+", default=[4, 8, 11])
parser.add_argument("--position", type=int, default=8)
parser.add_argument("--extent", type=float, default=4.0)
args = parser.parse_args()
out = ROOT / "natural_dct" / f"L{args.length}_t{'-'.join(map(str, args.targets))}"

device = torch.device("cuda")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model, _, _ = load_tensor_gpt("Elriggs/gpt2-swiglu-18l-9h-1152embd-v2", device)
model.requires_grad_(False)
pool = [t for t in read_texts(0, 520)[1] if len(tokenizer(t).input_ids) >= args.length]
anchors, held = pool[32:56], pool[56:]

L, R = torch.load(out / "dct_seed0.pt")
dct_l, dct_r = (F.normalize(x, dim=0).T for x in (L, R))
generator = torch.Generator().manual_seed(31)
span_l, span_r = (F.normalize(torch.linalg.qr(x).Q @ torch.randn(16, 16, generator=generator).to(device), dim=0).T for x in (L, R))
derangement = torch.roll(torch.arange(16), 1)
families = {"dct": (dct_l, dct_r), "mispaired": (dct_l, dct_r[derangement]), "span": (span_l, span_r)}

ids = tokenizer(held, return_tensors="pt", truncation=True, max_length=args.length).input_ids.to(device)
with torch.no_grad():
    start = model.state_before_block(ids, 8)
    block = model.transformer.h[8]
    values = block.lambdas[0] * start["values"] + block.lambdas[1] * start["initial_values"]
    attention, _ = block.attn(F.rms_norm(values, (1152,)), start["first_values"])
    inputs = F.rms_norm(values + attention, (1152,))[:, 1:].reshape(-1, 1152)

anchor_state = collect_middle_inputs(model, tokenizer, anchors, 8, args.length)
ops = [SlicedModel(model, tuple(s[c:c + 1] for s in anchor_state), 8, model.config.n_layer, [args.position]) for c in range(len(anchors))]
site = torch.zeros(args.length, device=device)
site[args.position] = 1
coords = np.linspace(-args.extent, args.extent, 41)


def logits(residual):
    return 30 * torch.tanh(model.lm_head(F.rms_norm(residual, (1152,))).float() / 30)


@torch.no_grad()
def disk(l, r, chunk=512):
    # perturbation_disk conventions: Gram-Schmidt the second axis, each axis in units of its own data SD
    e2 = F.normalize(r - (r @ l) * l, dim=0)
    sd = torch.stack([inputs @ l, inputs @ e2]).std(1)
    grid = torch.tensor(coords, dtype=torch.float32, device=device)
    a, b = (x.flatten() for x in torch.meshgrid(grid, grid, indexing="ij"))
    theta = (a[:, None] * sd[0] * l + b[:, None] * sd[1] * e2)[:, None] * site[None, :, None]
    distances = []
    for op in ops:
        y = torch.cat([logits(op(t, t, op.reference_values.expand(len(t), -1, -1), torch.zeros(len(t), 1152, device=device))[0])
                       for t in theta.split(chunk)])
        distances.append((y - y[len(y) // 2]).norm(dim=-1))
    return torch.stack(distances).median(0).values.view(len(coords), len(coords)).cpu().numpy(), (r @ l).item()


grids = {}
for family, (ls, rs) in families.items():
    for k in range(8):
        grids[family, k] = disk(ls[k], rs[k])
    print(f"done {family}", flush=True)
np.savez(out / f"levelsets_pos{args.position}_{args.extent:g}.npz", **{f"{f}_{k}": g for (f, k), (g, _) in grids.items()})

levels = np.median([g.max() for g, _ in grids.values()]) * np.array([0.25, 0.5, 0.75])
fig, axes = plt.subplots(len(families), 8, figsize=(17.6, 2.4 * len(families)), squeeze=False)
for row_axes, family in zip(axes, families):
    for k, ax in enumerate(row_axes):
        grid, cos = grids[family, k]
        image = ax.imshow(grid.T, origin="lower", extent=(coords[0], coords[-1], coords[0], coords[-1]), cmap="Greys",
                          vmin=0, vmax=levels[-1] / 0.75)
        ax.contour(coords, coords, grid.T, levels=levels, colors=["C0", "C1", "C3"])
        ax.set_title(f"{family} {k}  cos={cos:.2f}", fontsize=8)
        ax.tick_params(labelsize=6)
    row_axes[0].set_ylabel("r⊥ (data SD)", fontsize=8)
for ax in axes[-1]:
    ax.set_xlabel("l (data SD)", fontsize=8)
fig.colorbar(image, ax=axes, shrink=0.6, label="L2 logit distance (median over anchors)")
fig.suptitle(f"single-site push at position {args.position}, layer-8 MLP input; readout: logits at the same position; "
             f"contours at fixed logit distances {np.round(levels, 2).tolist()}", fontsize=9)
fig.savefig(out / f"levelsets_pos{args.position}_{args.extent:g}.png", dpi=120)
