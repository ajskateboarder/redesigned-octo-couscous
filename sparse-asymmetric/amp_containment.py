"""The gated subspace (mode_bound.md, Theorem 8) on a saved run: how well do trained fingerprints fit the
spiked–incoherent class, and what does containment give?

For V = the top-k eigenvectors of R = E[(P - P̄)²] (k from --ks):
    capacities      outside τ⊥ = max_{E, w ⊥ V} wᵀM_E w and cross τ× = max_{E, v ∈ V, w ⊥ V} vᵀM_E w, by search (alternate
                    E ← top-p rows of the quadratic / bilinear form, direction ← top eigen / singular vector); searches give
                    lower bounds on the true maxima
    isotropic null  the same search after rotating every row by an independent Haar rotation of V⊥ (V fixed): the
                    capacities of an isotropic remainder at this sample size
    class constants γ_N = √((2E‖P‖²_F + ¼)/(p(1-p))) and γ_× = √(E‖P‖²_F/(p(1-p))), upper bounds on the capacities of an
                    isotropic remainder N ⪯ P in the population (Theorem 8 (c))
    containment     ε = τ×² / (τ×² + (γ - τ⊥)²) and the count k/(1 - ε) at each γ, with the searched capacities
    inside V        orthonormal gates found by a self-gate search restricted to V (as in amp_bound_tightness.py)
    --nc-only       Corollary 8.1 instead: a greedy estimate of N_c, the number of orthonormal directions whose own share has
                    variance ≥ c = p(1-p)(γ/d)² (alternate shifted-free power steps s ← E[(sᵀDs) D s], D = P - P̄, projected off
                    the directions already found). Greedy gives a lower bound on N_c, not a certificate

MLPs followed only by the readout (gelu_1l --layer 0, gelu_2l --layer 1), Fisher metric as in amp_mode_bound.py.

Usage: python amp_containment.py <run .pt> [--ks 9,15,25,40] [--gamma 25,50,100] [--nc-only]
Output: downstream_amp/checks/amp_containment[_nc]_<run>.log, downstream_amp/mode_bound/containment[_nc]_<run>.pt
"""
import argparse
import math

import torch
from torch.func import jacrev, vmap

from amp_common import RUNS, from_run, log_to

parser = argparse.ArgumentParser()
parser.add_argument("run")
parser.add_argument("--ks", default="9,15,25,40", help="dimensions k of the candidate subspaces V")
parser.add_argument("--gamma", default="25,50,100", help="contrast thresholds γ (d·λ)")
parser.add_argument("--starts", type=int, default=32)
parser.add_argument("--iters", type=int, default=30)
parser.add_argument("--batch", type=int, default=32)
parser.add_argument("--nc-only", action="store_true", help="only the greedy N_c estimate of Corollary 8.1")
cli = parser.parse_args()
path, res, S_ = from_run(cli.run)
log_to(f"amp_containment{'_nc' if cli.nc_only else ''}_{path.stem}")
torch.set_grad_enabled(False)
Mo, L, dev, D, FIRST, TOPK, tail = S_.M, S_.L, S_.dev, S_.D, S_.FIRST, S_.TOPK, S_.args.tail
assert L == Mo.n_layer - 1, "G_ρ is computed in closed form: the MLP must be followed only by the readout"
parts = Mo.mlp_parts(L)
SLOPE = {"gelu": lambda u: 0.5 * (1 + torch.erf(u / math.sqrt(2))) + u * torch.exp(-u ** 2 / 2) / math.sqrt(2 * math.pi),
         "relu": lambda u: (u > 0).float(), "relu2": lambda u: 2 * u.clamp_min(0), "sq": lambda u: 2 * u}
slope = SLOPE[parts.get("act", parts["kind"])]
W = (S_.ROOT @ parts["w_in"].float()).contiguous()
Wout, bias = parts["W_out"].float(), parts["bias"]
GAMMAS = sorted(float(g) for g in cli.gamma.split(","))
KS = [int(k) for k in cli.ks.split(",")]
B, IT = cli.starts, cli.iters
g0 = torch.Generator(device=dev).manual_seed(0)
jac_head = vmap(jacrev(lambda h, idx: Mo.head(h[None, None], idx[None, None])[0, 0], chunk_size=128))


def fingerprints(R):
    # P_ρ = GᵀG / ‖G‖²_F as in amp_mode_bound.py, rows flattened to D² entries
    x, r = R.x[:, FIRST:].reshape(-1, D), R.r[:, FIRST:].reshape(-1, D)
    p, idx = R.p.reshape(-1, TOPK).float(), R.idx.reshape(-1, TOPK)
    X = torch.empty(len(x), D * D, device=dev)
    for s in range(0, len(x), cli.batch):
        sl = slice(s, s + cli.batch)
        u = parts["pre"](x[sl]).float()
        Lr = jac_head(r[sl] + parts["hidden"](x[sl]).float() @ Wout + bias, idx[sl]) @ Wout.T
        sp, sm = p[sl].sqrt(), p[sl].sum(-1, keepdim=True)
        c = (1 - (1 - sm).clamp_min(0).sqrt()) / sm
        BL = sp[..., None] * Lr
        BL = (BL - (c * sp)[..., None] * (sp[:, None, :] @ BL)) / math.sqrt(2)
        G = (BL * slope(u)[:, None, :]) @ W.T
        X[sl] = (G.transpose(1, 2) @ G / G.square().sum((1, 2))[:, None, None]).reshape(-1, D * D)
    return X


X = fingerprints(S_.fit)
N = len(X)
n_on = round(tail * N)
Pv = X.view(-1, D, D)
Pbar = X.mean(0).view(D, D)
purity = X.square().sum(1).mean().item()
pq = tail * (1 - tail)
print(f"{S_.args.model} L{L} ({path.name}): d = {D}, {N} fitting rows; p = {tail:g}; E‖P‖²_F = {purity:.4f}")


def xi(on):
    on = on.float()
    return on / on.sum(0) - (1 - on) / (1 - on).sum(0)


def top_event(q):
    on = torch.zeros_like(q, dtype=torch.bool)
    on.scatter_(0, q.topk(n_on, dim=0).indices, True)
    return on


def bilin(Vb, Wb):
    # vᵀ P_ρ w per row, for paired columns of Vb, Wb
    return torch.cat([torch.einsum("nde,db,eb->nb", Pv[i:i + 512], Vb, Wb) for i in range(0, N, 512)])


def event_mats(f):
    return (xi(top_event(f)).T @ X).view(-1, D, D)


def starts(Up):
    Wc = torch.cat([torch.linalg.eigh((Up.T @ Pbar @ Up).double())[1][:, -B // 2:].float(),
                    torch.randn(Up.shape[1], B - B // 2, device=dev, generator=g0)], 1)
    return torch.nn.functional.normalize(Wc, dim=0)


def outside_capacity(Up):
    Wc = starts(Up)
    for _ in range(IT):
        S = Up @ Wc
        Wc = torch.linalg.eigh((Up.T @ event_mats(bilin(S, S)) @ Up).double())[1][:, :, -1].float().T
    S = Up @ Wc
    return D * torch.einsum("db,bde,eb->b", S, event_mats(bilin(S, S)), S).max().item()


def cross_capacity(Uv, Up):
    a = torch.nn.functional.normalize(torch.randn(Uv.shape[1], B, device=dev, generator=g0), dim=0)
    bw = starts(Up)
    for _ in range(IT):
        U_, _, Vh = torch.linalg.svd(Uv.T @ event_mats(bilin(Uv @ a, Up @ bw)) @ Up)
        a, bw = U_[:, :, 0].T, Vh[:, 0, :].T
    Ms = event_mats(bilin(Uv @ a, Up @ bw))
    return D * torch.einsum("db,bde,eb->b", Uv @ a, Ms, Up @ bw).abs().max().item()


def gates_inside(Uv):
    # greedy orthonormal self-gates restricted to span(Uv); their contrasts
    found, A_ = [], []
    while len(found) < Uv.shape[1]:
        F = torch.stack(found, 1) if found else torch.zeros(D, 0, device=dev)
        Q = Uv @ Uv.T - F @ F.T
        S = torch.nn.functional.normalize(torch.cat([torch.linalg.eigh((Q @ Pbar @ Q).double())[1][:, -B // 2:].float(),
                                                     Q @ torch.randn(D, B - B // 2, device=dev, generator=g0)], 1), dim=0)
        for _ in range(IT):
            S = torch.linalg.eigh((Q @ event_mats(bilin(S, S)) @ Q).double())[1][:, :, -1].float().T
        A = D * torch.einsum("db,bde,eb->b", S, event_mats(bilin(S, S)), S)
        b = int(A.argmax())
        if A[b] < GAMMAS[0]:
            break
        s = S[:, b] - F @ (F.T @ S[:, b])
        found.append(s / s.norm())
        A_.append(A[b].item())
    return torch.tensor(A_)


if cli.nc_only:
    # Corollary 8.1: greedy orthonormal directions of largest own-share variance Var(sᵀPs) = E(sᵀDs)², D = P - P̄
    cmin = pq * (GAMMAS[0] / D) ** 2
    found, v = [], []
    while len(found) < D:
        F = torch.stack(found, 1) if found else torch.zeros(D, 0, device=dev)
        Q = torch.eye(D, device=dev) - F @ F.T
        S = torch.nn.functional.normalize(Q @ torch.cat([torch.linalg.eigh((Q @ Pbar @ Q).double())[1][:, -B // 2:].float(),
                                                         torch.randn(D, B - B // 2, device=dev, generator=g0)], 1), dim=0)
        for _ in range(2 * IT):
            q = bilin(S, S) - torch.einsum("db,de,eb->b", S, Pbar, S)
            Gs = sum(torch.einsum("nb,nde,eb->db", q[i:i + 512], Pv[i:i + 512] - Pbar, S) for i in range(0, N, 512))
            S = torch.nn.functional.normalize(Q @ Gs, dim=0)
        var = bilin(S, S).var(0)
        b = int(var.argmax())
        if var[b] < cmin:
            break
        s_ = S[:, b] - F @ (F.T @ S[:, b])
        found.append(s_ / s_.norm())
        v.append(bilin(found[-1][:, None], found[-1][:, None])[:, 0].var().item())
    v = torch.tensor(v)
    tight = RUNS / "mode_bound" / f"tightness_{path.stem}.pt"
    adm = torch.load(tight)["summary"] if tight.exists() else {}
    print("Corollary 8.1, greedy N_c (lower bound) vs gates the fingerprints admit (amp_bound_tightness.py): " + ", ".join(
        f"γ = {g:g}: N_c ≥ {int((v >= pq * (g / D) ** 2).sum())}, gates {adm.get(g, {}).get('m_self', '?')}" for g in GAMMAS))
    print(f"   variances / c(γ = {GAMMAS[0]:g}): " + " ".join(f"{x / cmin:.1f}" for x in v.tolist()))
    torch.save({"args": vars(cli), "variances": v, "c": {g: pq * (g / D) ** 2 for g in GAMMAS}},
               RUNS / "mode_bound" / f"containment_nc_{path.stem}.pt")
    raise SystemExit

EP2 = sum((Pv[i:i + 512] @ Pv[i:i + 512]).sum(0) for i in range(0, N, 512)) / N
Rv = torch.linalg.eigh((EP2 - Pbar @ Pbar).double())[1].flip(1).float()
del EP2
gN, gX = math.sqrt((2 * purity + 0.25) / pq), math.sqrt(purity / pq)
print(f"class constants (isotropic remainder, population, upper): γ_N = {gN:.1f}, γ_× = {gX:.1f}")
print(f"capacities in contrast units, searched; 'iso' = remainder Haar-rotated in V⊥; η = (real / iso)²")
print(f"   {'V':10s} | {'τ⊥':>5} {'iso':>4} {'η⊥':>4} | {'τ×':>5} {'iso':>4} {'η×':>4} | " +
      " | ".join(f"γ={g:g}: ε, k/(1-ε), inside V" for g in GAMMAS))
g_rot = torch.Generator(device=dev).manual_seed(11)
X0 = X.clone()
out = {"purity": purity, "gamma_N": gN, "gamma_x": gX, "rows": []}
for k in KS:
    Q = torch.linalg.qr(torch.cat([Rv[:, :k], torch.randn(D, D - k, device=dev, generator=g0)], 1))[0]
    Uv, Up = Q[:, :k], Q[:, k:]
    tp, tx = outside_capacity(Up), cross_capacity(Uv, Up)
    inside = gates_inside(Uv)
    for i in range(0, N, 256):                                    # isotropic null: Haar-rotate each row inside V⊥
        n = min(256, N - i)
        Qm, Rm = torch.linalg.qr(torch.randn(n, D - k, D - k, device=dev, generator=g_rot))
        Rot = torch.eye(D, device=dev).repeat(n, 1, 1)
        Rot[:, k:, k:] = Qm * torch.sign(torch.diagonal(Rm, dim1=1, dim2=2))[:, None, :]
        Pb = Q.T @ X0[i:i + n].view(-1, D, D) @ Q
        X[i:i + n] = (Q @ (Rot @ Pb @ Rot.transpose(1, 2)) @ Q.T).reshape(n, -1)
    tp0, tx0 = outside_capacity(Up), cross_capacity(Uv, Up)
    X.copy_(X0)
    cells = []
    for g in GAMMAS:
        if tp >= g:
            cells.append(f"γ={g:g}:    –      –   {int((inside >= g).sum()):3d}")
        else:
            eps = tx ** 2 / (tx ** 2 + (g - tp) ** 2)
            cells.append(f"γ={g:g}: {eps:4.2f} {k / (1 - eps):6.1f} {int((inside >= g).sum()):3d}")
    print(f"   R top-{k:<4d}| {tp:5.1f} {tp0:4.1f} {(tp / tp0) ** 2:4.0f} | {tx:5.1f} {tx0:4.1f} {(tx / tx0) ** 2:4.0f} | "
          + " | ".join(cells), flush=True)
    out["rows"].append({"k": k, "tau_perp": tp, "tau_x": tx, "tau_perp_iso": tp0, "tau_x_iso": tx0, "inside": inside})

(RUNS / "mode_bound").mkdir(exist_ok=True)
torch.save({"args": vars(cli), **out}, RUNS / "mode_bound" / f"containment_{path.stem}.pt")
print(f"saved {RUNS / 'mode_bound' / ('containment_' + path.stem + '.pt')}")
