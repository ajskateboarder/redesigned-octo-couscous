"""How tight are Theorems 5 and 7 (gate_bound.md) on a trained model? Both bound gates (E, s) over *arbitrary* events E of
prevalence p. For a fixed channel s the best such event is the top-p rows of q_s = sᵀPs, since
p(1-p)·(E[q | E] - E[q | Eᶜ]) = E[q 1_E] - p E[q] is maximised at fixed mass by the largest q. So the largest number of
orthonormal gates at contrast ≥ γ that the model's fingerprints admit can be searched for directly:

    self-gate search   greedy over orthonormal channels: alternate E ← top-p rows of q_s and s ← top eigenvector of
                       Q M_E Q (Q: off the channels already found); each step raises sᵀM_E s; keep the best start
    held-out           the channel's event on held-out rows is q_s above its fitting-row threshold
    bounds             Thm 7: m ≤ d/(pγ), spectral m ≤ max{m : (1/m)Σ_(i≤m) λ_i(P̄) ≥ pγ/d}; Thm 5: m ≤ κ/c,
                       spectral m ≤ max{m : (1/m)Σ_(i≤m) λ_i(K) ≥ c}, c = p(1-p)(γ/d)² (B = 1: orthonormal channels)
    Thm 7*             m·p(1-p)(γ/d) ≤ tr A(Y₀) and its spectral form, A(Y₀) = E[(1-p)(P - Y₀)₊ + p(P - Y₀)₋], Y₀ = θP̄
    budget accounting  per self-gate: p·Δ = E q - E[q | off] (Thm 7 drops E[q | off]); Σ_a p_a Δ_a ≤ E⟨P, F_on⟩ ≤ 1 with
                       F_on the projector on the channels open at the row (Thm 7′)
    opener class       for each self-gate channel, the best contrast from a single-neuron opener (top p of u_i) and
                       from a halfspace opener (top p of o·z, o the regression direction of q_s on whitened z)

MLPs followed only by the readout (gelu_1l --layer 0, gelu_2l --layer 1), where G_ρ is exact (as in amp_gate_bound.py).

Usage: python amp_bound_tightness.py <run .pt> [--gamma 25,50,100] [--starts 32] [--max-channels 400]
Output: downstream_amp/checks/amp_bound_tightness_<run>.log, downstream_amp/gate_bound/tightness_<run>.pt
"""
import argparse
import math
import time

import numpy as np
import torch
from torch.func import jacrev, vmap

from amp_common import RUNS, from_run, log_to

parser = argparse.ArgumentParser()
parser.add_argument("run")
parser.add_argument("--gamma", default="25,50,100", help="contrast thresholds γ (d·λ)")
parser.add_argument("--starts", type=int, default=32, help="starts per channel: half top eigenvectors of Q P̄ Q, half random")
parser.add_argument("--iters", type=int, default=30)
parser.add_argument("--max-channels", type=int, default=400)
parser.add_argument("--batch", type=int, default=32)
parser.add_argument("--freeze", choices=["none", "readout", "slopes"], default="none",
                    help="readout: P̃_ρ = W D_ρ Q̄ D_ρ Wᵀ / tr(·), Q̄ = E[Q_ρ / tr Q_ρ] with Q_ρ = (C^½L)ᵀ(C^½L) the output Gram, so "
                    "only the neurons' slopes D_ρ = diag φ'(u_ρ) vary across rows; slopes: D_ρ replaced by its mean, so only the "
                    "readout varies. Either is compared with the unfrozen channels of the same metric")
parser.add_argument("--metric", choices=["fisher", "identity"], default="fisher",
                    help="output metric on the top-k logits: fisher C = ½(diag p - ppᵀ) (downstream_amp.py's gain), or identity "
                    "on the centred logits (raw logit sensitivity, no next-token distribution); identity runs are compared with fisher")
cli = parser.parse_args()
path, res, S_ = from_run(cli.run)
BASE = "" if cli.metric == "fisher" else f"_metric{cli.metric}"
TAG = BASE + ("" if cli.freeze == "none" else f"_freeze{cli.freeze}")
log_to(f"amp_bound_tightness{TAG}_{path.stem}")
torch.set_grad_enabled(False)
Mo, L, dev, D, P_, FIRST, TOPK, tail = S_.M, S_.L, S_.dev, S_.D, S_.P_, S_.FIRST, S_.TOPK, S_.args.tail
assert L == Mo.n_layer - 1, "G_ρ is computed in closed form: the MLP must be followed only by the readout"
parts = Mo.mlp_parts(L)
SLOPE = {"gelu": lambda u: 0.5 * (1 + torch.erf(u / math.sqrt(2))) + u * torch.exp(-u ** 2 / 2) / math.sqrt(2 * math.pi),
         "relu": lambda u: (u > 0).float(), "relu2": lambda u: 2 * u.clamp_min(0), "sq": lambda u: 2 * u}
slope = SLOPE[parts.get("act", parts["kind"])]
W = (S_.ROOT @ parts["w_in"].float()).contiguous()
H = W.shape[1]
Wout, bias = parts["W_out"].float(), parts["bias"]
GAMMAS = sorted(float(g) for g in cli.gamma.split(","))
g0 = torch.Generator(device=dev).manual_seed(0)
jac_head = vmap(jacrev(lambda h, idx: Mo.head(h[None, None], idx[None, None])[0, 0], chunk_size=128))


def readout(x, r, p, idx):
    # u (pre-activations) and C^½L (k × H): the row's output effect of each neuron, C = ½(diag p - ppᵀ); with
    # --metric identity, the centred logit Jacobian
    u = parts["pre"](x).float()
    Lr = jac_head(r + parts["hidden"](x).float() @ Wout + bias, idx) @ Wout.T
    if cli.metric == "identity":
        return u, Lr - Lr.mean(1, keepdim=True)
    sp, sm = p.sqrt(), p.sum(-1, keepdim=True)
    c = (1 - (1 - sm).clamp_min(0).sqrt()) / sm
    BL = sp[..., None] * Lr
    return u, (BL - (c * sp)[..., None] * (sp[:, None, :] @ BL)) / math.sqrt(2)


def rows(R):
    return R.x[:, FIRST:].reshape(-1, D), R.r[:, FIRST:].reshape(-1, D), R.p.reshape(-1, TOPK).float(), R.idx.reshape(-1, TOPK)


def output_gram(R):
    # Q̄ = E[Q_ρ / tr Q_ρ], Q_ρ = (C^½L)ᵀ(C^½L)
    x, r, p, idx = rows(R)
    Q = torch.zeros(H, H, device=dev)
    for s in range(0, len(x), cli.batch):
        sl = slice(s, s + cli.batch)
        _, BL = readout(x[sl], r[sl], p[sl], idx[sl])
        BL = BL / BL.square().sum((1, 2)).sqrt()[:, None, None]
        Q += torch.einsum("nkh,nkg->hg", BL, BL)
    return Q / len(x)


def mean_slope(R):
    return slope(parts["pre"](rows(R)[0]).float()).mean(0)


def fingerprints(R, Qbar=None, Dbar=None):
    # P_ρ = GᵀG / ‖G‖²_F as in amp_gate_bound.py (or the frozen-readout P̃_ρ), rows flattened to D² entries; also u and x
    x, r, p, idx = rows(R)
    X = torch.empty(len(x), D * D, device=dev)
    for s in range(0, len(x), cli.batch):
        sl = slice(s, s + cli.batch)
        u, BL = readout(x[sl], r[sl], p[sl], idx[sl])
        if Qbar is None:
            G = (BL * (slope(u) if Dbar is None else Dbar.expand_as(u))[:, None, :]) @ W.T
            Pm = G.transpose(1, 2) @ G
        else:
            Bm = W * slope(u)[:, None, :]                        # read directions scaled by the row's slopes
            Pm = Bm @ Qbar @ Bm.transpose(1, 2)
        X[sl] = (Pm / torch.diagonal(Pm, dim1=1, dim2=2).sum(1)[:, None, None]).reshape(-1, D * D)
    return X, parts["pre"](x).float(), x


Qbar = output_gram(S_.fit) if cli.freeze == "readout" else None
Dbar = mean_slope(S_.fit) if cli.freeze == "slopes" else None
X, U, Xin = fingerprints(S_.fit, Qbar, Dbar)
Xh, Uh, Xinh = fingerprints(S_.held, Qbar, Dbar)
N, Nh = len(X), len(Xh)
n_on = round(tail * N)
print(f"{S_.args.model} L{L} ({path.name}): d = {D}, d_mlp = {H}, {N} fitting rows, {Nh} held-out rows; p = {tail:g}"
      + {"none": "", "readout": "; FROZEN READOUT (only slopes vary)", "slopes": "; FROZEN SLOPES (only the readout varies)"}[cli.freeze]
      + ("" if cli.metric == "fisher" else f"; metric {cli.metric}"))


def gains(Xr, S):
    # q_s(ρ) = sᵀP_ρ s for the columns s of S
    return torch.cat([torch.einsum("nde,dm,em->nm", Xr[i:i + 256].view(-1, D, D), S, S) for i in range(0, len(Xr), 256)])


def xi(on):
    on = on.float()
    return on / on.sum(0) - (1 - on) / (1 - on).sum(0)


def top_event(q, k=n_on):
    on = torch.zeros_like(q, dtype=torch.bool)
    on.scatter_(0, q.topk(k, dim=0).indices, True)
    return on


# --- kernel and mean fingerprint, for the bounds
Pbar = X.mean(0).view(D, D)
evP = torch.linalg.eigvalsh(Pbar.double()).flip(0)
Xm = X.mean(0)
gram = torch.zeros(N, N, device=dev, dtype=torch.float64)
for i in range(0, N, 1024):
    for j in range(i, N, 1024):
        blk = ((X[i:i + 1024] - Xm) @ (X[j:j + 1024] - Xm).T).double()
        gram[i:i + 1024, j:j + 1024] = blk
        gram[j:j + 1024, i:i + 1024] = blk.T
lamK = torch.linalg.eigvalsh(gram).flip(0).clamp_min(0) / N
del gram
kappa, purity = lamK.sum().item(), X.square().sum(1).mean().item()
print(f"κ = {kappa:.4f}, E‖P‖²_F = {purity:.4f}, ‖P̄‖²_F = {evP.square().sum():.4f}; λ₁(P̄) = {evP[0]:.4f}, "
      f"λ₁(K) = {lamK[0]:.4f}")


def spectral(lam, level):
    # largest m with (1/m) Σ_(i≤m) λ_i ≥ level
    avg = lam.cumsum(0) / torch.arange(1, len(lam) + 1, device=lam.device)
    ok = torch.nonzero(avg >= level)
    return int(ok[-1]) + 1 if len(ok) else 0


def majorant(Y0):
    # Thm 7*: A(Y₀) = E[(1-p)(P - Y₀)₊ + p(P - Y₀)₋]; every gate has p(1-p)·Δ ≤ sᵀA(Y₀)s, and Y₀ = 0 gives (1-p)P̄ (Thm 7)
    A = torch.zeros(D, D, device=dev, dtype=torch.float64)
    for i in range(0, N, 256):
        ev, V = torch.linalg.eigh(X[i:i + 256].view(-1, D, D).double() - Y0.double())
        A += torch.einsum("nij,nj,nkj->ik", V, torch.where(ev > 0, (1 - tail) * ev, -tail * ev), V)
    return torch.linalg.eigvalsh(A / N).flip(0)


THETAS = (0.0, 1.0, 2.0)
evA = {th: majorant(th * Pbar) for th in THETAS}
print("Thm 7* majorants A(θP̄): tr A = β = " + ", ".join(f"{evA[th].sum():.3f} (θ = {th:g})" for th in THETAS))


# --- greedy self-gate search over orthonormal channels
found, A_fit = [], []
t_min = GAMMAS[0] / D
t0 = time.time()
while len(found) < cli.max_channels:
    Uf = torch.stack(found, 1) if found else torch.zeros(D, 0, device=dev)
    Q = torch.eye(D, device=dev) - Uf @ Uf.T
    QPQ = Q @ Pbar @ Q
    k_eig = cli.starts // 2
    S = torch.cat([torch.linalg.eigh(QPQ.double())[1][:, -k_eig:].float(),
                   Q @ torch.randn(D, cli.starts - k_eig, device=dev, generator=g0)], 1)
    S = torch.nn.functional.normalize(S, dim=0)
    prev = None
    for it in range(cli.iters):
        on = top_event(gains(X, S))
        if prev is not None and (on == prev).all():
            break
        prev = on
        Ms = (xi(on).T @ X).view(-1, D, D)
        S = torch.linalg.eigh((Q @ Ms @ Q).double())[1][:, :, -1].float().T
    on = top_event(gains(X, S))
    Ms = (xi(on).T @ X).view(-1, D, D)
    A = D * torch.einsum("dm,mde,em->m", S, Ms, S)
    best = int(A.argmax())
    if A[best] < GAMMAS[0]:
        break
    s = S[:, best] - Uf @ (Uf.T @ S[:, best])
    found.append(s / s.norm())
    A_fit.append(A[best].item())
    if len(found) % 25 == 0:
        print(f"   ... {len(found)} channels, last contrast {A_fit[-1]:.1f} ({time.time() - t0:.0f} s)")
Sf = torch.stack(found, 1)
m_all = Sf.shape[1]
A_fit = torch.tensor(A_fit, device=dev)

# re-evaluate every channel's own best event on the final (orthonormalised) channels, fit and held-out
q, qh = gains(X, Sf), gains(Xh, Sf)
on = top_event(q)
thr = q.topk(n_on, dim=0).values[-1]
onh = qh >= thr
mu1, mu0 = (q * on).sum(0) / on.sum(0), (q * ~on).sum(0) / (~on).sum(0)
A_self = D * (mu1 - mu0)
A_held = D * ((qh * onh).sum(0) / onh.sum(0).clamp_min(1) - (qh * ~onh).sum(0) / (~onh).sum(0))
p_held = onh.float().mean(0)
print(f"\nself-gate search: {m_all} orthonormal channels with fitted contrast ≥ {GAMMAS[0]:g} "
      f"(max {A_self.max():.0f}); held-out prevalence of their events: median {p_held.median():.3f}")

# --- opener class: single neurons and halfspaces on the same channels
thrU = U.quantile(1 - tail, dim=0)
A_neuron = torch.cat([D * (xi(U[:, i:i + 512] >= thrU[i:i + 512]).T @ q) for i in range(0, H, 512)]).max(0).values
Z = (Xin - S_.MU) @ S_.IROOT
o = torch.nn.functional.normalize(Z.T @ (q - q.mean(0)), dim=0)          # regression direction of q_s on whitened z
A_half = D * (xi(top_event(Z @ o)) * q).sum(0)

print(f"\n{'γ':>5} | {'self (fit)':>10} {'self (held)':>11} | {'Thm 7':>6} {'Thm 7 sp.':>9} | {'Thm 7* tr':>9} {'Thm 7* sp.':>10} | "
      f"{'Thm 5 κ':>8} {'Thm 5 sp.':>9} | {'neuron':>6} {'halfspace':>9} | Σ p·Δ (of 1)  Σ Var / (m c)")
summary = {}
for g in GAMMAS:
    t, c = g / D, tail * (1 - tail) * (g / D) ** 2
    sel = A_self >= g
    m, mh = int(sel.sum()), int((sel & (A_held >= g)).sum())
    b7, b7s = D / (tail * g), spectral(evP, tail * t)
    b7q = min(evA[th].sum().item() / (tail * (1 - tail) * t) for th in THETAS)
    b7qs = min(spectral(evA[th], tail * (1 - tail) * t) for th in THETAS)
    b5, b5s = kappa / c, spectral(lamK, c)
    used = (tail * (mu1 - mu0))[sel].sum().item()
    var_ratio = (q[:, sel].var(0) / c).mean().item() if m else float("nan")
    summary[g] = dict(m_self=m, m_held=mh, thm7=b7, thm7_spectral=b7s, thm7q=b7q, thm7q_spectral=b7qs, thm5=b5,
                      thm5_spectral=b5s, m_neuron=int((A_neuron >= g).sum()), m_half=int((A_half >= g).sum()),
                      budget_used=used)
    print(f"{g:5.0f} | {m:10d} {mh:11d} | {b7:6.0f} {b7s:9d} | {b7q:9.0f} {b7qs:10d} | {b5:8.0f} {b5s:9d} | "
          f"{int((A_neuron >= g).sum()):6d} {int((A_half >= g).sum()):9d} | {used:12.3f}  {var_ratio:10.2f}")

# --- Theorem 7 accounting per self-gate (γ = smallest): where each channel's share E[q] goes
sel = A_self >= GAMMAS[0]
Eq = q.mean(0)
print(f"\nThm 7 accounting at γ = {GAMMAS[0]:g} ({int(sel.sum())} channels): p·Δ / E[q] median "
      f"{(tail * (mu1 - mu0) / Eq)[sel].median():.2f} (the rest is the off-mean E[q | off]); on/off ratio median "
      f"{(mu1 / mu0)[sel].median():.1f}")
on_sel = on[:, sel].float()
cap_on = (on_sel * q[:, sel]).sum(1)                                            # ⟨P_ρ, F_on(ρ)⟩, orthonormal channels
print(f"Thm 7′: Σ_a p_a Δ_a = {(tail * (mu1 - mu0))[sel].sum():.3f} ≤ E⟨P, F_on⟩ = {cap_on.mean():.3f} ≤ 1; open channels "
      f"per row: mean {on_sel.sum(1).mean():.1f}, max {int(on_sel.sum(1).max())}")
top5 = torch.linalg.eigh(Pbar.double())[1][:, -5:].float()
print(f"first channels: contrast fit / held / best neuron / best halfspace; d·E[q]; share in the top 5 eigenvectors of P̄")
for a in range(min(12, m_all)):
    print(f"   {a:3d}: {A_self[a]:6.1f} {A_held[a]:6.1f} {A_neuron[a]:6.1f} {A_half[a]:6.1f}   {D * Eq[a]:5.2f}   "
          f"{(top5.T @ Sf[:, a]).square().sum():.2f}")

REF = f"tightness{BASE}_{path.stem}.pt" if cli.freeze != "none" else f"tightness_{path.stem}.pt"
if TAG and (RUNS / "gate_bound" / REF).exists():
    # same gates? the reference channels' self-gate contrast under these fingerprints, and the overlap of the two sets
    real = torch.load(RUNS / "gate_bound" / REF)
    Sr = real["channels"].to(dev)
    qr = gains(X, Sr)
    onr = top_event(qr)
    A_r = D * ((qr * onr).sum(0) / onr.sum(0) - (qr * ~onr).sum(0) / (~onr).sum(0))
    cos = (Sr.T @ Sf).abs()
    sv = torch.linalg.svdvals(Sr.T @ Sf)
    print(f"\nreference channels ({REF}, {Sr.shape[1]}): contrast there → here")
    print("   " + ", ".join(f"{a:.0f} → {b:.0f}" for a, b in zip(real["A_self"].tolist(), A_r.tolist())))
    print(f"   best |cos| of each reference channel with a channel here: {np.round(cos.max(1).values.cpu().numpy(), 2).tolist()}")
    print(f"   principal cosines between the two channel spans: {np.round(sv.cpu().numpy(), 2).tolist()}")

out = RUNS / "gate_bound"
out.mkdir(exist_ok=True)
torch.save({"args": vars(cli), "model": S_.args.model, "d": D, "H": H, "kappa": kappa, "purity": purity, "evP": evP.cpu(),
            "lamK": lamK[:2048].cpu(), "evA": {th: e.cpu() for th, e in evA.items()}, "channels": Sf.cpu(), "A_self": A_self.cpu(), "A_held": A_held.cpu(),
            "A_neuron": A_neuron.cpu(), "A_half": A_half.cpu(), "summary": summary}, out / f"tightness{TAG}_{path.stem}.pt")
print(f"\nsaved {out / ('tightness' + TAG + '_' + path.stem + '.pt')}")
