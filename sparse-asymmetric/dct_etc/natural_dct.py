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
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import mannwhitneyu
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import GPT2Tokenizer

from bilinear_attention_pr_experiment import AttentionRegularizedBilinearDCT, SlicedModel
from intermediate_mlp_pr_experiment import collect_middle_inputs, read_texts
from tensor_model import load_tensor_gpt

parser = argparse.ArgumentParser()
parser.add_argument("--length", type=int, default=12)
parser.add_argument("--targets", type=int, nargs="+", default=[4, 8, 11])
args = parser.parse_args()
targets = list(args.targets)
out = ROOT / "natural_dct" / f"L{args.length}_t{'-'.join(map(str, targets))}"
out.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model, _, _ = load_tensor_gpt("Elriggs/gpt2-swiglu-18l-9h-1152embd-v2", device)
model.requires_grad_(False)
# truncate prompts with >= length tokens so no position is padding
pool = [t for t in read_texts(0, 520)[1] if len(tokenizer(t).input_ids) >= args.length]
train, anchors, held = pool[:32], pool[32:56], pool[56:]

cache = out / "dct_seed0.pt"
if cache.exists():
    L, R = torch.load(cache)
else:
    state = collect_middle_inputs(model, tokenizer, train, 8, args.length)
    delta = SlicedModel(model, state, 8, 12, targets)
    zero = torch.zeros(1152, device=device)
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
        clean = delta(zero, zero, state[0], torch.zeros_like(state[0]))[0]
    with sdpa_kernel(SDPBackend.MATH):
        torch.manual_seed(0)
        dct = AttentionRegularizedBilinearDCT(16, 0.03, 1.0)
        dct.fit(delta, state[0], clean, 8, 16, 100)
    L, R = dct.L.detach().float(), dct.R.detach().float()
    torch.save((L, R), cache)
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
sigma = (inputs @ F.normalize(torch.randn(1152, 256, generator=torch.Generator().manual_seed(37)).to(device), dim=0)).std(0).median().item()
centered = inputs - inputs.mean(0)

held_state = collect_middle_inputs(model, tokenizer, held, 8, args.length)
natural = SlicedModel(model, held_state, 8, 12, targets)
zero = torch.zeros(1152, device=device)
with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
    natural_targets = torch.cat([natural(zero, zero, v, torch.zeros(len(v), 1152, device=device))[0] for v in held_state[0].split(64)])
spread = (natural_targets - natural_targets.mean(0)).norm(dim=-1).square().mean().sqrt().item()
del held_state, natural

anchor_state = collect_middle_inputs(model, tokenizer, anchors, 8, args.length)
ops = [SlicedModel(model, tuple(s[c:c + 1] for s in anchor_state), 8, 12, targets) for c in range(len(anchors))]
target_mask = torch.zeros(args.length, device=device)
target_mask[targets] = 1
masks = {"all": torch.ones(args.length, device=device), "targets": target_mask, "others": 1 - target_mask}

coords = np.linspace(-4.0, 4.0, 17)
center = len(coords) // 2


@torch.no_grad()
def plane(l, r, mask, chunk=512):
    grid = torch.tensor(coords * sigma, dtype=torch.float32, device=device)
    a, b = (x.flatten() for x in torch.meshgrid(grid, grid, indexing="ij"))
    theta = (a[:, None] * l + b[:, None] * r)[:, None] * mask[None, :, None]
    inter, total = [], []
    for op in ops:
        y = torch.cat([op(t, t, op.reference_values.expand(len(t), -1, -1), torch.zeros(len(t), 1152, device=device))[0]
                       for t in theta.split(chunk)]).view(len(coords), len(coords), -1)
        total.append((y - y[center, center]).square().sum(-1))
        inter.append((y - y[:, center:center + 1] - y[center:center + 1] + y[center, center]).square().sum(-1))
    return torch.stack(inter).cpu().numpy(), torch.stack(total).cpu().numpy()


def token_rms(grids, a, b):
    points = np.clip(np.stack([a, b], 1), coords[0], coords[-1])
    return np.array([np.sqrt(RegularGridInterpolator((coords, coords), g)(points).mean()) for g in grids]) / spread


print(f"prompts kept {len(pool)} (train {len(train)}, anchors {len(anchors)}, held-out {len(held)}), targets {targets}")
print(f"sigma {sigma:.3f}, natural target spread {spread:.2f}, tokens {len(inputs)}")
print("token-weighted RMS over the natural cloud, / natural spread; median over factors [IQR]")
print(f"{'family':10s} {'push':8s} | {'response':>22s} | {'AND part':>22s} | {'AND share':>9s}")
results, clouds, totals = {}, {}, {}
for family, (ls, rs) in families.items():
    a_tok = (centered @ ls.T / sigma).cpu().numpy()
    b_tok = (centered @ rs.T / sigma).cpu().numpy()
    clouds[family] = a_tok, b_tok
    for name, mask in masks.items():
        rows = []
        for k, (l, r) in enumerate(zip(ls, rs)):
            inter, total = plane(l, r, mask)
            if name == "all" and k < 8:
                totals[family, k] = np.median(np.sqrt(total), 0)
            rows.append([np.median(token_rms(total, a_tok[:, k], b_tok[:, k])), np.median(token_rms(inter, a_tok[:, k], b_tok[:, k]))])
        rows = np.array(rows)
        results[family, name] = rows
        share = np.median(rows[:, 1] ** 2 / rows[:, 0] ** 2)
        print(f"{family:10s} {name:8s} | {np.median(rows[:, 0]):7.3f} [{np.percentile(rows[:, 0], 25):.3f}-{np.percentile(rows[:, 0], 75):.3f}] | "
              f"{np.median(rows[:, 1]):7.3f} [{np.percentile(rows[:, 1], 25):.3f}-{np.percentile(rows[:, 1], 75):.3f}] | {share:9.3f}", flush=True)
for family in ("mispaired", "span"):
    print(f"AND part, all positions, dct vs {family}: Mann-Whitney p = "
          f"{mannwhitneyu(results['dct', 'all'][:, 1], results[family, 'all'][:, 1]).pvalue:.2g}")
np.savez(out / "results.npz", **{f"{f}_{n}": v for (f, n), v in results.items()})

fig, axes = plt.subplots(len(families), 8, figsize=(17.6, 2.3 * len(families)), squeeze=False)
for row_axes, family in zip(axes, families):
    a_tok, b_tok = clouds[family]
    for k, ax in enumerate(row_axes):
        grid = totals[family, k]
        ax.contour(coords, coords, grid.T / grid.max(), levels=[0.25, 0.5, 0.75], colors=["C0", "C1", "C3"])
        ax.scatter(a_tok[:, k], b_tok[:, k], s=1, c="k", alpha=0.15, linewidths=0)
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_aspect("equal")
        ax.set_title(f"{family} {k}", fontsize=8)
        ax.tick_params(labelsize=6)
    row_axes[0].set_ylabel("r (σ)", fontsize=8)
for ax in axes[-1]:
    ax.set_xlabel("l (σ)", fontsize=8)
fig.tight_layout()
fig.savefig(out / "planes_4_tokens.png", dpi=120)
