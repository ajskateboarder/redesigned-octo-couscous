import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for path in (str(ROOT.parent), str(ROOT), "/tmp/fsec"):
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np
import torch
from scipy.stats import spearmanr
from torch.func import jvp
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


def to_mlp8_input(op, values):
    # exact path from the block-k residual to the layer-8 MLP input (blocks k..7 in full, block 8 up to its MLP norm)
    initial, first = op.initial_values, op.first_values
    for rel, block in enumerate(op.blocks[:8 - op.source_layer + 1]):
        values = block.lambdas[0] * values + block.lambdas[1] * initial
        attention, first, _ = op.attn(block.attn, F.rms_norm(values, (1152,)), first)
        values = values + attention
        if op.source_layer + rel == 8:
            return F.rms_norm(values, (1152,))
        values = values + SlicedModel.mlp(block.mlp, F.rms_norm(values, (1152,)))


state8 = collect_middle_inputs(model, tokenizer, anchors, 8, length)
ops8 = [SlicedModel(model, tuple(s[c:c + 1] for s in state8), 8, 12, targets) for c in range(len(anchors))]
mlp8_clean = torch.cat([to_mlp8_input(op, op.reference_values) for op in ops8])


@torch.no_grad()
def transported_plane(tangent_l, tangent_r, chunk=2048):
    # linear transport of the residual push to the layer-8 MLP input, then the exact model from the layer-8 MLP on (both branches)
    grids = []
    for op, tl, tr in zip(ops8, tangent_l, tangent_r):
        theta = (a_unit * extent)[:, None, None] * tl + (b_unit * extent)[:, None, None] * tr
        y = torch.cat([op(t, t, op.reference_values.expand(len(t), -1, -1), torch.zeros(len(t), 1152, device=device))[0] for t in theta.split(chunk)])
        grids.append((y - y[len(y) // 2]).norm(dim=-1))
    return torch.stack(grids).view(-1, size, size).cpu().numpy()


saved = np.load(out / "plateau_p.npz")
rows = {}
for k in (8, 6, 4):
    state = collect_middle_inputs(model, tokenizer, anchors, k, length)
    ops = [SlicedModel(model, tuple(s[c:c + 1] for s in state), k, 12, targets) for c in range(len(anchors))]
    with torch.no_grad():
        residual = model.state_before_block(ids, k)["values"][:, 1:].reshape(-1, 1152).float()
        check = (torch.cat([to_mlp8_input(op, op.reference_values) for op in ops]) - mlp8_clean).abs().max().item()
    print(f"block {k}: path to layer-8 MLP input reproduces clean input to {check:.1e}", flush=True)
    for family, (ls, rs) in families.items():
        for i, (l, r) in enumerate(zip(ls, rs)):
            sd = torch.stack([(residual @ l).std(), (residual @ r).std()])
            tangents = []
            for direction, scale in ((l, sd[0]), (r, sd[1])):
                push = (scale * direction).expand(1, length, 1152).clone()
                tangents.append(torch.cat([jvp(lambda v: to_mlp8_input(op, v), (op.reference_values,), (push,))[1] for op in ops]))
            tl, tr = tangents
            basis = torch.linalg.qr(torch.stack([l, r], 1)).Q
            in_span = lambda t: ((t @ basis).square().sum(-1) / t.square().sum(-1)).median().item()
            rows[k, family, i] = {
                "A": float(saved[f"b | residual @{k} | {family}"][i]),
                "B": p_of(transported_plane(tl, tr)),
                "span_l": in_span(tl), "span_r": in_span(tr),
                "cos_l": F.cosine_similarity(tl, l, dim=-1).median().item(), "cos_r": F.cosine_similarity(tr, r, dim=-1).median().item(),
            }
    del state, ops
    print(f"done block {k}", flush=True)

print("\nA = actual push at block k; B = same push transported linearly to the layer-8 MLP input, exact model from there")
print("in-span = fraction of the transported push lying in span(l_i, r_i); cos = alignment with its own direction (medians over anchors and positions)")
for family in families:
    print(f"\n{family}")
    for k in (8, 6, 4):
        r_ = [rows[k, family, i] for i in range(16)]
        get = lambda key: np.array([x[key] for x in r_])
        A, B = get("A"), get("B")
        print(f"  block {k}: p A {np.nanmedian(A):.2f} (p<1 {np.sum(A < 1):2d}/16) | p B {np.nanmedian(B):.2f} (p<1 {np.sum(B < 1):2d}/16) | "
              f"A−B {np.nanmedian(A - B):+.2f} | in-span l {np.median(get('span_l')):.2f} r {np.median(get('span_r')):.2f} | "
              f"cos l {np.median(get('cos_l')):.2f} r {np.median(get('cos_r')):.2f}")
    for k in (6, 4):
        A_shift = np.array([rows[k, family, i]["A"] - rows[8, family, i]["A"] for i in range(16)])
        B_shift = np.array([rows[k, family, i]["B"] - rows[8, family, i]["B"] for i in range(16)])
        align = np.array([min(rows[k, family, i]["span_l"], rows[k, family, i]["span_r"]) for i in range(16)])
        keep = np.isfinite(A_shift) & np.isfinite(B_shift)
        print(f"  block {k} vs 8, across factors: Spearman(A shift, B shift) {spearmanr(A_shift[keep], B_shift[keep]).statistic:+.2f}, "
              f"Spearman(A shift, in-span) {spearmanr(A_shift[keep], align[keep]).statistic:+.2f}")
    if family == "dct":
        for k in (6, 4):
            print(f"  per factor block {k}: A", np.round([rows[k, family, i]["A"] for i in range(16)], 2).tolist())
            print(f"  per factor block {k}: B", np.round([rows[k, family, i]["B"] for i in range(16)], 2).tolist())
np.save(out / "transport_p.npy", {f"{k}|{f}|{i}": v for (k, f, i), v in rows.items()}, allow_pickle=True)
