import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT.parent), str(ROOT)]

import numpy as np
import torch
from datasets import load_dataset
from scipy.sparse.csgraph import connected_components
from torch.nn import functional as F
from transformers import GPT2Tokenizer

from tensor_model import apply_rotary_emb, load_tensor_gpt

device = torch.device("cuda")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model, _, _ = load_tensor_gpt("Elriggs/gpt2-swiglu-18l-9h-1152embd-v2", device)
model.requires_grad_(False)
blocks = model.transformer.h[8:12]
out = ROOT / "natural_dct" / "bigram"
length, targets, sizes, draws, rank, threshold = 12, [4, 8, 11], [4, 8, 16, 32], range(4), 16, 0.8
sites, eps = 64, torch.finfo(torch.float32).eps
texts = [t.strip() for t in load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"]
         if t.strip() and not t.strip().startswith("=")]
ids = torch.tensor([s[:length] for s in (tokenizer(t).input_ids for t in texts) if len(s) >= length][:2048], device=device)
components = {"norm", "attention", "mlp8", "mlps"}
configurations = {"only MLP 8": components - {"mlp8"}, "only MLPs 9-11": components - {"mlps"},
                  "only attention": components - {"attention"}, "only norms": components - {"norm"},
                  "without norms": {"norm"}, "without attention": {"attention"}}


@torch.no_grad()
def start(batch):
    s = model.state_before_block(batch, 8)
    return s["values"].float(), s["initial_values"].float(), s["first_values"].float()


def recurring_factors(prefix):
    # cached fits on 16 independent bigram samples; a factor recurs if linked (|cos u|·|cos l|·|cos r| > 0.8) across ≥ 8 fits
    fits = [{k: F.normalize(v, dim=0) for k, v in torch.load(out / f"{prefix}n{n}_draw{d}.pt").items()} for n in sizes for d in draws]
    similarity = lambda a, b: (a["U"].T @ b["U"]).abs() * (a["L"].T @ b["L"]).abs() * (a["R"].T @ b["R"]).abs()
    links = np.zeros((len(fits) * rank,) * 2, dtype=bool)
    for a in range(len(fits)):
        for b in range(a + 1, len(fits)):
            links[a * rank:(a + 1) * rank, b * rank:(b + 1) * rank] = (similarity(fits[a], fits[b]) > threshold).numpy()
    groups = {}
    for flat, c in enumerate(connected_components(links, directed=False)[1]):
        groups.setdefault(c, []).append(divmod(flat, rank))
    recurring = []
    for g in sorted(groups.values(), key=lambda g: -len({f for f, _ in g})):
        if len({f for f, _ in g}) >= len(fits) // 2:
            score = [sum(similarity(fits[f], fits[h])[i, j].item() for h, j in g if h != f) for f, i in g]
            f, i = g[int(np.argmax(score))]
            recurring.append({k: v[:, i] for k, v in fits[f].items()})
    return recurring


def attention(attn, x, first, pattern=None):
    b, s, d = x.shape
    h, hd = attn.n_head, attn.head_dim
    value = attn.c_v(x).view(b, s, h, hd)
    value = (1 - attn.lamb) * value + attn.lamb * first.view_as(value)
    if pattern is None:
        query, key = attn.c_q(x).view(b, s, h, hd), attn.c_k(x).view(b, s, h, hd)
        cos, sin = attn.rotary(query)
        query = apply_rotary_emb(F.rms_norm(query, (hd,)), cos, sin).transpose(1, 2)
        key = apply_rotary_emb(F.rms_norm(key, (hd,)), cos, sin).transpose(1, 2)
        causal = torch.ones(s, s, device=x.device, dtype=torch.bool).tril()
        pattern = (query @ key.transpose(-1, -2) / hd ** 0.5).masked_fill(~causal, float("-inf")).softmax(-1)
    return attn.c_proj((pattern @ value.transpose(1, 2)).transpose(1, 2).reshape(b, s, d)), pattern


def forward(values, initial, first, theta_l=0.0, theta_r=0.0, frozen=(), clean=None):
    # blocks 8-11; frozen components reuse the clean run's norm scales, attention patterns or MLP gate activations
    record = {}
    for k, block in enumerate(blocks):
        values = block.lambdas[0] * values + block.lambdas[1] * initial
        record[k, "attention scale"] = s = clean[k, "attention scale"] if "norm" in frozen else torch.rsqrt(values.square().mean(-1, keepdim=True) + eps)
        out, record[k, "pattern"] = attention(block.attn, values * s, first, clean[k, "pattern"] if "attention" in frozen else None)
        values = values + out
        record[k, "mlp scale"] = s = clean[k, "mlp scale"] if "norm" in frozen else torch.rsqrt(values.square().mean(-1, keepdim=True) + eps)
        x = values * s
        tl, tr = (theta_l, theta_r) if k == 0 else (0.0, 0.0)
        if ("mlp8" if k == 0 else "mlps") in frozen:
            gate = clean[k, "gate"]
        else:
            left = block.mlp.left(x + tl)
            gate = left * torch.sigmoid(left)
        record[k, "gate"] = gate
        values = values + block.mlp.Down(gate * block.mlp.right(x + tr)) + block.mlp.Down_bias
        if k == 0:
            source = x
    return values[:, targets].mean(1), source, record


with torch.no_grad():
    check = ids[:8]
    reference = model.state_before_block(check, 12)["values"][:, targets].float().mean(1)
    assert torch.allclose(forward(*start(check))[0], reference, atol=1e-3 * reference.abs().max().item())


def ablate(x, direction, mean, positions):
    # move the direction's coordinate to its corpus mean at one position per row
    return -F.one_hot(positions, length).float()[..., None] * ((x @ direction - mean)[..., None] * direction)


def second_order(method, factor, means, state, positions, extra=None, frozen=(), clean=None):
    # the factor's per-token second-order push at the layer-12 targets; extra = perturbation of the token's own input
    values, initial, first = state
    if method == "asymmetric":
        u, l, r = factor
        x = forward(values, initial, first)[1]
        extra = torch.zeros_like(x) if extra is None else extra
        tl, tr = ablate(x + extra, l, means[0], positions), ablate(x + extra, r, means[1], positions)
        t = lambda a, b: forward(values, initial, first, extra + a, extra + b, frozen, clean)[0]
        return t(tl, tr) - t(tl, 0.0) - t(0.0, tr) + t(0.0, 0.0)
    u, v = factor
    base = values if extra is None else values + extra
    delta = ablate(base, v, means, positions)
    t = lambda z: forward(z, initial, first, frozen=frozen, clean=clean)[0]
    return (t(base + delta) + t(base - delta)) / 2 - t(base)


@torch.no_grad()
def labels(method, parts, means, batch=96):
    # second-order push along u at every position > 0 of every context
    out = []
    for chunk in ids.split(batch):
        state = tuple(t.repeat_interleave(length - 1, 0) for t in start(chunk))
        positions = torch.arange(1, length, device=device).repeat(len(chunk))
        out.append(second_order(method, parts, means, state, positions) @ parts[0])
    return torch.cat(out)


def participation(matrix):
    energy = torch.linalg.svdvals(matrix).square()
    return (energy.sum().square() / energy.square().sum()).item(), (energy[0] / energy.sum()).item()


def top_sites(method, factor):
    if method == "asymmetric":
        u, l, r = (factor[k].to(device) for k in "ULR")
        with torch.no_grad():
            x_all = torch.cat([forward(*start(chunk))[1][:, 1:] for chunk in ids.split(256)]).reshape(-1, 1152)
        means, parts = ((x_all @ l).mean(), (x_all @ r).mean()), (u, l, r)
    else:
        u, v = factor["U"].to(device), factor["L"].to(device)
        means, parts = (start(ids)[0][:, 1:] @ v).mean(), (u, v)
    y = labels(method, parts, means)
    side = max((1, -1), key=lambda s: (s * y).topk(int(0.02 * len(y))).values.mean())
    chosen = (side * y).topk(sites).indices
    return parts, means, side, chosen // (length - 1), chosen % (length - 1) + 1


def analyse(method, factor):
    parts, means, side, rows, positions = top_sites(method, factor)
    u = parts[0]
    own = torch.linalg.qr(torch.stack(parts[1:], 1)).Q
    state = start(ids[rows])
    with torch.no_grad():
        clean = forward(*state)[2]
        full = second_order(method, parts, means, state, positions)
        kept = {name: ((second_order(method, parts, means, state, positions, frozen=f, clean=clean) @ u) * side).sum().item()
                / ((full @ u) * side).sum().item() for name, f in configurations.items()}
    along_u = ((full @ u).square() / full.square().sum(-1)).median().item()
    output_dims, output_top = participation(full)

    token = torch.zeros(sites, 1152, device=device, requires_grad=True)
    extra = F.one_hot(positions, length).float()[..., None] * token[:, None, :]
    (side * (second_order(method, parts, means, state, positions, extra=extra) @ u)).sum().backward()
    gradient = token.grad.detach()
    reading = (forward(*state)[1] if method == "asymmetric" else state[0])[torch.arange(sites), positions].detach()
    total = gradient.square().sum(-1)
    in_own = ((gradient @ own).square().sum(-1) / total).median().item()
    along_self = ((gradient * F.normalize(reading, dim=-1)).sum(-1).square() / total).median().item()
    rest = gradient - (gradient @ own) @ own.T
    input_dims, _ = participation(F.normalize(rest, dim=-1))
    return kept, along_u, output_dims, output_top, in_own, along_self, input_dims


def main():
    print(f"{sites} strongest WikiText sites per factor (stronger side of u); second-order push = the factor's per-token effect at layer 12")
    print("kept = share of that push along u that remains when only the named component stays nonlinear (others frozen at their clean values)\n")
    summary = {}
    for method, prefix in (("asymmetric", ""), ("quadratic", "quadratic_")):
        summary[method] = []
        for k, factor in enumerate(recurring_factors(prefix)):
            kept, along_u, output_dims, output_top, in_own, along_self, input_dims = analyse(method, factor)
            summary[method].append([*kept.values(), along_u, output_dims, output_top, in_own, along_self, input_dims])
            print(f"{method} #{k}: kept " + ", ".join(f"{name} {value:+.2f}" for name, value in kept.items()))
            print(f"  output at layer 12: share along u {along_u:.2f}; across sites, top direction {output_top:.2f} of energy, "
                  f"effective number of directions {output_dims:.1f}")
            print(f"  input gradient: share in the factor's own input direction(s) {in_own:.2f}, along the token's own activation {along_self:.2f}; "
                  f"rest spans {input_dims:.1f} effective directions")
        print()

    names = [*configurations, "share along u", "output directions", "top output direction", "input share in own directions",
             "input share along own activation", "other input directions"]
    for method, rows in summary.items():
        print(f"{method} (median over factors): " + ", ".join(f"{n} {v:.2f}" for n, v in zip(names, np.median(np.array(rows), axis=0))))


if __name__ == "__main__":
    main()
