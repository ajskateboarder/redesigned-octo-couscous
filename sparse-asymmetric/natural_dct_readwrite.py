import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for path in (str(ROOT.parent), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np
import torch
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
families = {"dct": (dct_l, dct_r), "span": (span_l, span_r)}
grids = np.load(out / f"levelsets_pos{args.position}_{args.extent:g}.npz")
coords = np.linspace(-args.extent, args.extent, grids["dct_0"].shape[0])

full = [tokenizer(t).input_ids for t in held]
ids = torch.tensor([f[:args.length] for f in full], device=device)
with torch.no_grad():
    start = model.state_before_block(ids, 8)
    block = model.transformer.h[8]
    values = block.lambdas[0] * start["values"] + block.lambdas[1] * start["initial_values"]
    attention, _ = block.attn(F.rms_norm(values, (1152,)), start["first_values"])
    inputs = F.rms_norm(values + attention, (1152,))[:, 1:].reshape(-1, 1152)
centered = inputs - inputs.mean(0)
sites = [(p, t) for p in range(len(held)) for t in range(1, args.length)]

anchor_state = collect_middle_inputs(model, tokenizer, anchors, 8, args.length)
ops = [SlicedModel(model, tuple(s[c:c + 1] for s in anchor_state), 8, model.config.n_layer, [args.position]) for c in range(len(anchors))]
site = torch.zeros(args.length, device=device)
site[args.position] = 1


def logits(residual):
    return 30 * torch.tanh(model.lm_head(F.rms_norm(residual, (1152,))).float() / 30)


@torch.no_grad()
def logit_shift(theta):
    theta = torch.stack([torch.zeros_like(theta), theta])[:, None] * site[None, :, None]
    shifts = [logits(op(theta, theta, op.reference_values.expand(2, -1, -1), torch.zeros(2, 1152, device=device))[0]) for op in ops]
    return torch.stack([y[1] - y[0] for y in shifts]).mean(0)


def show(token_ids):
    return " ".join(repr(tokenizer.decode([i])) for i in token_ids)


def context(p, t, width=5):
    before = tokenizer.decode(full[p][max(0, t - width):t])
    return f"{before}[{tokenizer.decode([full[p][t]])}]".replace("\n", " ")


for family, (ls, rs) in families.items():
    print(f"\n=== {family} ===")
    for k in range(8):
        l, r = ls[k], rs[k]
        e2 = F.normalize(r - (r @ l) * l, dim=0)
        a, b = centered @ l, centered @ e2
        sd = torch.stack([a, b]).std(1)
        grid = grids[f"{family}_{k}"]
        i, j = np.unravel_index(grid.argmax(), grid.shape)
        corner = torch.tensor([coords[i], coords[j]], dtype=torch.float32, device=device)
        direction = F.normalize(corner, dim=0)
        score = (a / sd[0]) * direction[0] + (b / sd[1]) * direction[1]
        read = score.argsort(descending=True)[:8].cpu()

        shift = logit_shift(corner[0] * sd[0] * l + corner[1] * sd[1] * e2)
        promoted = shift.topk(12).indices.cpu()
        demoted = (-shift).topk(10).indices.cpu()

        print(f"\n{family} {k}: corner (l, r⊥) = ({corner[0]:+.1f}, {corner[1]:+.1f}) SD")
        print("  reads  :", " | ".join(context(*sites[n]) for n in read[:8]))
        print("  writes+:", show(promoted.tolist()))
        print("  writes-:", show(demoted.tolist()))
