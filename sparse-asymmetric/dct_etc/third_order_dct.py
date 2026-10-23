"""Second- vs third-order asymmetric DCT at the MLP-8 branch inputs: do both collapse onto single units?

Slice 8→12, readout = mean over target positions [4, 8, 11] of the layer-12 residual, unit pushes θ_l (gate-branch input)
and θ_r (up-branch input) at every position, as in the post. Per context x, with Δ the readout change,
    order 2:   y₂(x) = D_l D_r Δ          (the post's objective)
    order 3:   y₃(x) = D_l D_l D_r Δ      (adds the terms where l passes a second curvature in series, e.g. g''[h''[l,l], J r])
and the objective Σ_x (u·y(x))² is fitted by a census of independent random starts (no orthogonalisation), updating
(u, l, r) <- normalize(∇) with an SS-HOPM shift on l for order 3 (the objective is quartic in l).
Each unit i is also scored at its own rows (l = gate row v̂_i, r = up row ŵ_i) with the best u (top singular value² of
the context × D matrix of y), and compared with random directions.

Usage: python third_order_dct.py [--contexts 16] [--starts 32] [--iters 40]
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT.parent), str(ROOT)]

import numpy as np
import torch
from scipy.sparse.csgraph import connected_components
from torch.func import grad_and_value, jvp, vmap
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from bigram_mechanism import device, model, out, start, targets, tokenizer
from bilinear_attention_pr_experiment import AttentionTargetDelta, SlicedModel

parser = argparse.ArgumentParser()
parser.add_argument("--contexts", type=int, default=16)
parser.add_argument("--starts", type=int, default=32)
parser.add_argument("--iters", type=int, default=40)
parser.add_argument("--chunk", type=int, default=8)
args = parser.parse_args()
torch.set_float32_matmul_precision("high")
length, D = 12, 1152
mlp = model.transformer.h[8].mlp
V, W = (F.normalize(w.weight.float(), dim=1) for w in (mlp.Left, mlp.Right))      # unit gate rows, unit up rows

next_ids, next_probs = torch.load(out / "chain.pt")
generator = torch.Generator(device=device).manual_seed(1000 + 10 * args.contexts)
token = torch.full((args.contexts,), tokenizer.eos_token_id, device=device)
sequence = []
for _ in range(length):
    token = next_ids[token, torch.multinomial(next_probs[token], 1, generator=generator)[:, 0]]
    sequence.append(token)
values, initial, first = start(torch.stack(sequence, 1))
target = AttentionTargetDelta(SlicedModel(model, (values, initial, first), 8, 12, targets))
clean = torch.zeros(args.contexts, D, device=device)


def response(order, l, r, x, y):
    # D_l^(order-1) D_r of the readout change, per context
    zero = torch.zeros_like(l)
    g = lambda a, b: target(a, b, x, y)
    d_l = lambda a, b: jvp(lambda a_: g(a_, b), (a,), (l,))[1]
    d_ll = lambda a, b: jvp(lambda a_: d_l(a_, b), (a,), (l,))[1]
    inner = d_l if order == 2 else d_ll
    return jvp(lambda b: inner(zero, b), (zero,), (r,))[1].float()


def make_step(order):
    def step(u, l, r):
        objective = lambda u, l, r: (response(order, l, r, values, clean) @ u).square().sum()
        (gu, gl, gr), value = grad_and_value(objective, argnums=(0, 1, 2))(u, l, r)
        return gu, gl, gr, value
    return vmap(step, in_dims=(1, 1, 1), out_dims=(1, 1, 1, 0), chunk_size=args.chunk)


def fit(order):
    gen = torch.Generator(device=device).manual_seed(order)
    U, L, R = (F.normalize(torch.randn(D, args.starts, device=device, generator=gen), dim=0) for _ in range(3))
    step = make_step(order)
    begin = time.time()
    for it in range(args.iters):
        with sdpa_kernel(SDPBackend.MATH):
            gU, gL, gR, E = step(U, L, R)
        # ∇_l of a form homogeneous of degree 2(order-1) in l is 2(order-1)E·l along l; the shift makes the l update monotone
        shift = 0.0 if order == 2 else 4.0 * E
        new = [F.normalize(gU, dim=0), F.normalize(gL + shift * L, dim=0), F.normalize(gR, dim=0)]
        drift = torch.stack([(a * b).sum(0).abs() for a, b in zip(new, (U, L, R))]).min(0).values
        U, L, R = new
        if it % 10 == 9 or it == args.iters - 1:
            print(f"  order {order} sweep {it + 1}: min |cos| to previous {drift.min().item():.5f}, median {drift.median().item():.5f} "
                  f"({time.time() - begin:.0f} s)", flush=True)
    return U, L, R, E


@torch.no_grad()
def best_u_energy(order, Ls, Rs, chunk=64):
    # max_u Σ_x (u·y(x))² = top singular value² of the (contexts × D) matrix of responses
    energies = []
    one = vmap(lambda l, r: response(order, l, r, values, clean), in_dims=(0, 0), chunk_size=args.chunk)
    for s in range(0, len(Ls), chunk):
        with sdpa_kernel(SDPBackend.MATH):
            Y = one(Ls[s:s + chunk], Rs[s:s + chunk])
        energies.append(torch.linalg.svdvals(Y)[:, 0].square())
    return torch.cat(energies)


results = {}
for order in (2, 3):
    print(f"\norder {order}: {args.starts}-start census on {args.contexts} bigram contexts", flush=True)
    U, L, R, E = fit(order)
    Lf, Rf = L.T, R.T
    gate_match, up_match = (Lf @ V.T).abs(), (Rf @ W.T).abs()
    best_gate, best_up = gate_match.max(1), up_match.max(1)
    same_unit = (gate_match * up_match).max(1)
    sim = ((Lf @ Lf.T).abs() * (Rf @ Rf.T).abs()).cpu().numpy()
    label = connected_components(sim > 0.8, directed=False)[1]
    sizes = sorted(np.bincount(label), reverse=True)
    units = best_u_energy(order, V, W)
    gen = torch.Generator(device=device).manual_seed(5)
    random = best_u_energy(order, F.normalize(torch.randn(256, D, device=device, generator=gen), dim=1),
                           F.normalize(torch.randn(256, D, device=device, generator=gen), dim=1))
    fitted = best_u_energy(order, Lf, Rf)
    top_units = torch.argsort(units, descending=True)
    print(f"order {order}: {sum(s >= 2 for s in sizes)} attractors, basin sizes {sizes[:10]}")
    print(f"  end points: l = a single gate row (|cos| > 0.9) in {(best_gate.values > 0.9).sum().item()}/{args.starts}, r = a single "
          f"up row in {(best_up.values > 0.9).sum().item()}/{args.starts}, both rows of the same unit (product > 0.8) in "
          f"{(same_unit.values > 0.8).sum().item()}/{args.starts}; median best |cos| l {best_gate.values.median().item():.2f}, "
          f"r {best_up.values.median().item():.2f}")
    print(f"  units the end points land on: {sorted(set(same_unit.indices[same_unit.values > 0.8].tolist()))}; top units by direct "
          f"scan: {top_units[:8].tolist()}")
    print(f"  energy (best u), relative to the median random pair: best unit {(units.max() / random.median()).item():.3g}, "
          f"10th unit {(units[top_units[9]] / random.median()).item():.3g}, median unit {(units.median() / random.median()).item():.3g}, "
          f"best random {(random.max() / random.median()).item():.3g}, best end point {(fitted.max() / random.median()).item():.3g}, "
          f"median end point {(fitted.median() / random.median()).item():.3g}")
    share = units / units.sum()
    print(f"  concentration over the 4608 units: top unit holds {share.max().item():.1%} of the summed unit energy, top 10 "
          f"{share[top_units[:10]].sum().item():.1%}; participation ratio {(1 / share.square().sum()).item():.0f}")
    results[order] = {"L": Lf.cpu(), "R": Rf.cpu(), "U": U.T.cpu(), "units": units.cpu(), "random": random.cpu(),
                      "fitted": fitted.cpu(), "label": label}
torch.save(results, out / f"third_order_n{args.contexts}.pt")
r2, r3 = results[2]["units"], results[3]["units"]
rank = lambda x: torch.argsort(torch.argsort(x))
print(f"\nSpearman between the unit rankings of order 2 and order 3: "
      f"{np.corrcoef(rank(r2).numpy(), rank(r3).numpy())[0, 1]:.3f}")
