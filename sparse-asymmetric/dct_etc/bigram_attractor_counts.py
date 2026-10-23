import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT.parent), str(ROOT)]

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.sparse.csgraph import connected_components
from torch.func import grad, vmap
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from bigram_mechanism import device, forward, model, out, start, targets, tokenizer
from bilinear_attention_pr_experiment import AttentionTargetDelta, SlicedModel
from bilinear_quadratic_experiment import ordered_cross_hessian_outputs
from dct import directional_hessian_outputs

runs, length, sizes, default = 64, 12, [4, 8, 16, 32], 0.8
steps = {"Amp": 300, "quadratic DCT": 30, "asymmetric DCT": 60}
forms = {"Amp": "v → v", "quadratic DCT": "v·v → u", "asymmetric DCT": "l·r → u"}
stems = {"Amp": "amp_whitened", "quadratic DCT": "quadratic", "asymmetric DCT": "asymmetric"}
next_ids, next_probs = torch.load(out / "chain.pt")
mlp = model.transformer.h[8].mlp
left_w, right_w, down_w = (w.weight.float() for w in (mlp.Left, mlp.Right, mlp.Down))


def sample(n):
    # the draw-0 bigram sample of n contexts used by the 16-factor fits
    generator = torch.Generator(device=device).manual_seed(1000 + 10 * n)
    token = torch.full((n,), tokenizer.eos_token_id, device=device)
    sequence = []
    for _ in range(length):
        token = next_ids[token, torch.multinomial(next_probs[token], 1, generator=generator)[:, 0]]
        sequence.append(token)
    return torch.stack(sequence, 1)


def random_directions(count, seed):
    generator = torch.Generator(device=device).manual_seed(seed)
    return [F.normalize(torch.randn(1152, runs, device=device, generator=generator), dim=0) for _ in range(count)]


with torch.no_grad():
    # whitening metric from the model's own bigram chain (2048 contexts), shared by every data size
    generator = torch.Generator(device=device).manual_seed(7)
    token = torch.full((2048,), tokenizer.eos_token_id, device=device)
    sequence = []
    for _ in range(length):
        token = next_ids[token, torch.multinomial(next_probs[token], 1, generator=generator)[:, 0]]
        sequence.append(token)
    pool = torch.cat([forward(*start(chunk))[1][:, 1:] for chunk in torch.stack(sequence, 1).split(256)]).reshape(-1, 1152)
    covariance = torch.cov(pool.T)
    covariance += 1e-3 * covariance.trace() / 1152 * torch.eye(1152, device=device)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    root = eigenvectors @ torch.diag(eigenvalues.sqrt()) @ eigenvectors.T
    inverse_root = eigenvectors @ torch.diag(eigenvalues.rsqrt()) @ eigenvectors.T


def fit_dct(method, n, batch=8, chunk=16):
    values, initial, first = start(sample(n))
    operator = SlicedModel(model, (values, initial, first), 8, 12, targets)
    zero = torch.zeros(1152, device=device)
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
        clean = operator(zero, zero, values, torch.zeros_like(values))[0]
    if method == "asymmetric DCT":
        target = AttentionTargetDelta(operator)
        objective = lambda u, l, r, x, y: 0.5 * (ordered_cross_hessian_outputs(target, l, r, x, y).float() @ u).square().sum()
        parts = random_directions(3, 1)
    else:
        residual = lambda theta, x, y: operator(zero, zero, x + theta, y)[0]
        objective = lambda u, v, x, y: directional_hessian_outputs(residual, v, v, x, y).float().sum(0) @ u
        parts = random_directions(2, 2)
    gradients = vmap(grad(objective, argnums=tuple(range(len(parts)))), in_dims=(1,) * len(parts) + (None, None),
                     out_dims=(1,) * len(parts), chunk_size=chunk)
    for _ in range(steps[method]):
        # independent single-factor power iterations: no orthogonalization across runs
        updates = [torch.zeros_like(p) for p in parts]
        for s in range(0, n, batch):
            with sdpa_kernel(SDPBackend.MATH):
                for total, g in zip(updates, gradients(*parts, values[s:s + batch], clean[s:s + batch])):
                    total += g
        parts = [F.normalize(g, dim=0) for g in updates]
    return parts


def fit_amp(n, tail=0.1, temperature=0.1):
    with torch.no_grad():
        z = forward(*start(sample(n)))[1][:, 1:].reshape(-1, 1152)
        gate, up = mlp.left(z), mlp.right(z)
        sigmoid = torch.sigmoid(gate)
        swish, slope_up = gate * sigmoid, sigmoid * (1 + gate * (1 - sigmoid)) * up
    centered = z - z.mean(0)
    w = random_directions(1, 3)[0].T.clone().requires_grad_()
    optimizer = torch.optim.Adam([w], lr=0.01)
    for _ in range(steps["Amp"]):
        d = F.normalize(w, dim=-1)
        # whitened metric: push along Σ^½ d, read along Σ^-½ d; transmission = readᵀ J push per token
        push, read = d @ root, d @ inverse_root
        passed = slope_up @ ((push @ left_w.T) * (read @ down_w)).T + swish @ ((push @ right_w.T) * (read @ down_w)).T
        standard = centered @ read.T
        standard = (standard - standard.mean(0)) / standard.std(0)
        weight = torch.sigmoid((standard - standard.detach().quantile(1 - tail, dim=0)) / temperature)
        amp = (weight * passed).sum(0) / weight.sum(0) - ((1 - weight) * passed).sum(0) / (1 - weight).sum(0)
        optimizer.zero_grad()
        (-amp.sum()).backward()
        optimizer.step()
    return [F.normalize(w.detach(), dim=-1).T]


def similarity(method, parts):
    if method == "Amp":
        return (parts[0].T @ parts[0]).abs()
    u, l = parts[0], parts[1]
    r = parts[2] if method == "asymmetric DCT" else l
    return (u.T @ u).abs() * (l.T @ l).abs() * (r.T @ r).abs()


def basins(matrix, threshold):
    _, label = connected_components((matrix > threshold).cpu().numpy(), directed=False)
    return sorted(np.bincount(label), reverse=True)


results = {}
for n in sizes:
    for method in steps:
        path = out / f"attractors_{stems[method]}_n{n}.pt"
        if not path.exists():
            parts = fit_amp(n) if method == "Amp" else fit_dct(method, n)
            torch.save([p.cpu() for p in parts], path)
        results[method, n] = similarity(method, torch.load(path))
        sizes_ = basins(results[method, n], default)
        print(f"N = {n:2d} {method:15s}: {sum(s >= 2 for s in sizes_)} attractors reached by ≥ 2 runs, "
              f"{len(sizes_)} distinct end points; basin sizes {sizes_[:10]}", flush=True)

slate, muted, coral = "#4b5d6e", "#6b7c8e", "#ee7853"
plt.rcParams["font.family"] = "serif"
offsets = np.linspace(-0.24, 0.24, len(sizes))
fig, ax = plt.subplots(figsize=(7, 4.8))
for k, method in enumerate(steps):
    counts = [sum(s >= 2 for s in basins(results[method, n], default)) for n in sizes]
    median = ax.hlines(np.median(counts), k - 0.33, k + 0.33, color=coral, lw=3.5, zorder=2)
    dots = ax.scatter(k + offsets, counts, s=42, color=slate, alpha=0.75, edgecolors="none", zorder=3)
    for x, y, n in zip(k + offsets, counts, sizes):
        ax.annotate(str(n), (x, y), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=8, color=muted)
ax.set_xticks(range(len(steps)), [f"{method}\n({forms[method]})" for method in steps])
ax.set_xlim(-0.6, len(steps) - 0.4)
ax.set_ylim(-0.6, 13.6)
ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(2))
ax.tick_params(colors=muted, labelsize=12)
ax.spines[["top", "right"]].set_visible(False)
ax.spines[["left", "bottom"]].set_color(muted)
ax.set_ylabel(f"Attractors ({runs} random starts)", fontsize=14, color="#33475b")
ax.set_axisbelow(True)
ax.grid(True, axis="y", color="#e5e5e5", lw=0.8)
ax.legend([dots, median], ["fits (label: bigram contexts)", "median over data sizes"], loc="upper left", fontsize=11)
fig.tight_layout()
fig.savefig(out / "attractor_counts.png", dpi=300)
print(f"saved {out / 'attractor_counts.png'}")
