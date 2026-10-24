"""Dense-gate bound (gate_bound.md): checks of the rank lemma, the per-opener and total gate-count bounds, on a saved
run's rows, for an MLP followed only by the readout (gelu_1l --layer 0, gelu_2l --layer 1), where G_ρ is exact.

    G_ρ = C_ρ^½ J_ρ Σ^½ = F_ρ Wᵀ     W = Σ^½ W_in (D × H), column i: neuron i's whitened read direction w_i
                                     F_ρ = C_ρ^½ L_ρ diag φ'(u_ρ) (k × H), L_ρ = ∂(top-k logits)/∂h · W_outᵀ,
                                     C_ρ = ½(diag p - p pᵀ) (so ‖G_ρ s‖² is downstream_amp.py's gain of s)
    P_ρ = G_ρᵀG_ρ / Z_ρ, Z_ρ = ‖G_ρ‖²_F           fingerprint (tr P = 1); d·sᵀP s is g(s)/τ with τ exact
    opener S (k neurons)  on = top --tail of rows by Σ_{i∈S} standardised pre-activation u_i (threshold from fitting rows)
    M_S = E[P | on] - E[P | off]                   gate direction: eigenvalue λ with d·|λ| ≥ γ (contrast units)
    E_S = Δ E[G_RᵀG_R / Z], G_R = G - F_S W_Sᵀ     what is left of M_S once every term touching w_S is removed
    Q_S                                            projector off span(w_S); Q M Q = Q E Q
Bounds checked (numbers as in gate_bound.md):
    Prop 1   per opener, unconditional: N_±(γ) ≤ d·Σ_±λ / γ ≤ d/γ (positive part of M has trace ≤ 1)
    Thm 3/3' per opener: N_±(γ) ≤ r + N_±(Q M Q, γ), r = rank w_S; so ≤ r if ‖Q M Q‖ < γ/d (implied by ‖E_S‖ < γ/d)
    Thm 5    all gates: m · p(1-p)(γ/d)² ≤ B · Σ_{i≤m} λ_i(K), K the covariance of P, B the frame bound of {s_a s_aᵀ}
Steps (gate_bound.md §3): 1 kernel spectrum; 2 ‖E_S‖, ‖Q M Q‖ vs γ/d; 3 gate-direction counts vs 2k; 4 read-direction overlap;
5 Fisher-mass shares, principal angles, B; 6 nulls (random row events, random neuron sets); 9 held-out replication,
window bootstrap, matrix-Bernstein radius. Openers: every single neuron, greedy k-neuron sets, the given sets, and the
run's fitted openers o (event: top --tail of c(o)) split on the given sets.

Usage: python amp_gate_bound.py <run .pt> [--sets 251,1608;251,1608,1942,690] [--opener-sets a0;a1;...] [--gamma 25,50,100]
Output: downstream_amp/checks/amp_gate_bound_<run>.log, downstream_amp/gate_bound/<run>.pt
"""
import argparse
import math

import numpy as np
import torch
from scipy.sparse.csgraph import connected_components
from torch.func import jacrev, vmap

from amp_common import RUNS, from_run, log_to, openers

parser = argparse.ArgumentParser()
parser.add_argument("run")
parser.add_argument("--sets", default="", help="neuron sets to test as openers, ';'-separated comma lists")
parser.add_argument("--opener-sets", default="", help="neuron set S for each of the run's attractors, ';'-separated")
parser.add_argument("--gamma", default="25,50,100", help="contrast thresholds γ (d·λ)")
parser.add_argument("--greedy", type=int, default=4, help="largest k of the greedy neuron-set search")
parser.add_argument("--seeds", type=int, default=8, help="greedy searches started from the best single neurons")
parser.add_argument("--nulls", type=int, default=64)
parser.add_argument("--boot", type=int, default=100, help="window-bootstrap resamples")
parser.add_argument("--batch", type=int, default=32)
cli = parser.parse_args()
path, res, S_ = from_run(cli.run)
log_to(f"amp_gate_bound_{path.stem}")
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
GAMMAS = [float(g) for g in cli.gamma.split(",")]
g0 = torch.Generator(device=dev).manual_seed(0)
jac_head = vmap(jacrev(lambda h, idx: Mo.head(h[None, None], idx[None, None])[0, 0], chunk_size=128))


def flat(R):
    return (R.x[:, FIRST:].reshape(-1, D), R.r[:, FIRST:].reshape(-1, D), R.p.reshape(-1, TOPK).float(),
            R.idx.reshape(-1, TOPK), torch.arange(R.n, device=dev).repeat_interleave(P_))


def factors(x, r, p, idx):
    u = parts["pre"](x).float()
    Lr = jac_head(r + parts["hidden"](x).float() @ Wout + bias, idx) @ Wout.T
    # C^½ = (I - c √p √pᵀ) diag √p / √2 with c = (1 - √(1 - Σp)) / Σp
    sp, sm = p.sqrt(), p.sum(-1, keepdim=True)
    c = (1 - (1 - sm).clamp_min(0).sqrt()) / sm
    BL = sp[..., None] * Lr
    BL = (BL - (c * sp)[..., None] * (sp[:, None, :] @ BL)) / math.sqrt(2)
    F_ = BL * slope(u)[:, None, :]
    G = F_ @ W.T
    return u, F_, G, G.square().sum((1, 2))


class Fingerprints:
    def __init__(self, R):
        self.x, self.r, self.p, self.idx, self.win = flat(R)
        self.N = len(self.x)
        self.u = parts["pre"](self.x).float()
        self.X = torch.empty(self.N, D * D, device=dev)
        self.Z = torch.empty(self.N, device=dev)
        for s, sl in self.batches():
            _, _, G, Z = factors(*self.at(sl))
            self.X[sl] = (G.transpose(1, 2) @ G / Z[:, None, None]).reshape(-1, D * D)
            self.Z[sl] = Z

    def batches(self):
        for s in range(0, self.N, cli.batch):
            yield s, slice(s, s + cli.batch)

    def at(self, sl):
        return self.x[sl], self.r[sl], self.p[sl], self.idx[sl]


def xi(on):
    on = on.float()
    return on / on.sum(0) - (1 - on) / (1 - on).sum(0)


def mats(Fp, Xi):
    return (Xi.T @ Fp.X).view(-1, D, D)


def eigs(Ms, chunk=128):
    return torch.cat([torch.linalg.eigvalsh(Ms[i:i + chunk].double()).float() for i in range(0, len(Ms), chunk)])


fit, held = Fingerprints(S_.fit), Fingerprints(S_.held)
mu_u, sd_u = fit.u.mean(0), fit.u.std(0)
print(f"{S_.args.model} L{L} ({path.name}): d = {D}, d_mlp = {H}, {fit.N} fitting rows ({fit.N // P_} windows), "
      f"{held.N} held-out rows; tail {tail:g}; γ/d for γ = {GAMMAS}: {[round(g / D, 4) for g in GAMMAS]}")

# --- sanity: ‖G s‖² is downstream_amp.py's gain, d·sᵀPs is g/τ with τ replaced by Z/d
sv = torch.nn.functional.normalize(torch.randn(D, 4, device=dev, generator=g0), dim=0)
g_pipe = S_.sweep(S_.held, sv)[1].reshape(4, -1)
g_here = torch.cat([(factors(*held.at(sl))[2] @ sv).square().sum(1) for _, sl in held.batches()]).T
tau_ratio = (S_.tau_h.flatten() / (held.Z / D))
print(f"sanity: ‖G s‖² vs downstream_amp gain, max rel. error {((g_here - g_pipe).abs() / g_pipe.abs().clamp_min(1e-9)).max():.2e}; "
      f"τ (8 random pushes) / (Z/d): median {tau_ratio.median():.3f}, 10-90% {tau_ratio.quantile(.1):.3f}-{tau_ratio.quantile(.9):.3f}")

# --- step 1: kernel. K = Cov(P) on Sym(d) (Frobenius); its nonzero spectrum is the centred Gram's / N
Xm = fit.X.mean(0)
gram = torch.zeros(fit.N, fit.N, device=dev, dtype=torch.float64)
for i in range(0, fit.N, 1024):
    for j in range(i, fit.N, 1024):
        blk = ((fit.X[i:i + 1024] - Xm) @ (fit.X[j:j + 1024] - Xm).T).double()
        gram[i:i + 1024, j:j + 1024] = blk
        gram[j:j + 1024, i:i + 1024] = blk.T
lamK, gvec = torch.linalg.eigh(gram)
lamK, g1 = lamK.flip(0).clamp_min(0) / fit.N, gvec[:, -1].float()
V1 = sum(g1[i:i + 1024] @ (fit.X[i:i + 1024] - Xm) for i in range(0, fit.N, 1024))
V1 = (V1 / V1.norm()).view(D, D)                            # top eigenvector of K (a symmetric matrix)
del gram, gvec
kappa = lamK.sum().item()
Pbar = Xm.view(D, D)
evP = torch.linalg.eigvalsh(Pbar.double()).flip(0)
pr_rows = 1 / fit.X.square().sum(1)                       # effective rank of each row's fingerprint
print(f"\n[1] kernel: κ = E‖P - P̄‖²_F = {kappa:.4f} (≤ 1); λ₁(K) = {lamK[0]:.4f}, λ₁/κ = {lamK[0] / kappa:.3f}; "
      f"top-10 λ(K)/κ {np.round((lamK[:10] / kappa).cpu().numpy(), 3).tolist()}; participation ratio of K "
      f"{kappa ** 2 / lamK.square().sum():.1f}")
print(f"    rows: ‖P_ρ‖²_F median {fit.X.square().sum(1).median():.3f} (effective rank 1/‖P‖²_F median {pr_rows.median():.1f}); "
      f"λ₁(P_ρ) median {torch.cat([torch.linalg.eigvalsh(fit.X[i:i + 256].view(-1, D, D))[:, -1] for i in range(0, fit.N, 256)]).median():.3f}")
ev_V1 = torch.linalg.eigvalsh(V1.double()).flip(0)
print(f"    top eigenvector V₁ of K: eigenvalues (as a matrix) {np.round(ev_V1[:3].cpu().numpy(), 2).tolist()} .. "
      f"{np.round(ev_V1[-3:].cpu().numpy(), 2).tolist()}; share of ‖V₁‖²_F in its top eigenvalue {ev_V1.abs().max() ** 2:.2f}")
print(f"    P̄: λ₁ {evP[0]:.3f}, top-5 {np.round(evP[:5].cpu().numpy(), 3).tolist()}, effective rank {1 / evP.square().sum():.1f}")


def thm5(c_over_B):
    # largest m with (1/m) Σ_{i≤m} λ_i(K) ≥ c/B
    avg = lamK.cumsum(0) / torch.arange(1, len(lamK) + 1, device=dev)
    ok = torch.nonzero(avg >= c_over_B)
    return int(ok[-1]) + 1 if len(ok) else 0


def thm5b(c, B=1.0):
    # Thm 5': if λ_{j+1}(K) < c, every gate has Σ_{i≤j} ⟨ssᵀ, V_i⟩² ≥ β_j = (c - λ_{j+1}) / (λ₁ - λ_{j+1}), so m ≤ B·j/β_j
    lk = lamK[:1024].double().cpu().numpy()
    if c > lk[0]:
        return 0.0, 0
    js = [j for j in range(1, len(lk)) if lk[j] < c]
    return min(((B * j * (lk[0] - lk[j]) / (c - lk[j]), j) for j in js), default=(math.inf, 0))


for g in GAMMAS:
    c = tail * (1 - tail) * (g / D) ** 2
    print(f"    Thm 5 at γ = {g:g} (p = {tail:g}, c = {c:.2e}): m ≤ B·κ/c = {kappa / c:.1f}·B; spectral form, B = 1: m ≤ {thm5(c)}, "
          f"B = 2: m ≤ {thm5(c / 2)}; Thm 5' (B = 1): m ≤ {thm5b(c)[0]:.1f} (j = {thm5b(c)[1]}); unconditional (κ ≤ 1): {1 / c:.0f}·B")

# --- noise floor: random row events of prevalence tail (by window, so row dependence is kept), and matrix Bernstein
nw = fit.N // P_
on_rand = torch.rand(fit.N, cli.nulls, device=dev, generator=g0) < tail
ev_rand = eigs(mats(fit, xi(on_rand)))
eps = ev_rand.abs().max(1).values
n_on = int(tail * fit.N)
v = Pbar.norm(2).item()
delta = 0.05
bern = lambda n, v: math.sqrt(2 * v * math.log(2 * D / delta) / n) + 2 * math.log(2 * D / delta) / (3 * n)
print(f"\nnoise floor (random row events, {cli.nulls}): ‖M‖ median {eps.median():.4f}, max {eps.max():.4f} "
      f"(d·‖M‖: {D * eps.median():.1f} / {D * eps.max():.1f}); matrix Bernstein radius (95%, i.i.d. rows, v ≈ ‖P̄‖ = {v:.3f}): "
      f"on-mean {bern(n_on, v):.4f}, off-mean {bern(fit.N - n_on, v):.4f}, with windows as units ({nw // 10} on) {bern(nw // 10, v):.4f}")

print(f"    so ‖M̂ - M‖ ≤ {bern(n_on, v) + bern(fit.N - n_on, v):.4f} (d·: {D * (bern(n_on, v) + bern(fit.N - n_on, v)):.1f}) "
      f"for i.i.d. rows, {bern(nw // 10, v) + bern(nw - nw // 10, v):.4f} (d·: {D * (bern(nw // 10, v) + bern(nw - nw // 10, v)):.1f}) "
      f"counting one row per window")


def count(ev, t):
    return (ev >= t).sum(-1), (ev <= -t).sum(-1)


# --- single-neuron openers (chunks of neurons): M_i, E_i = M_i - (a_i w_iᵀ + w_i a_iᵀ) + b_i w_i w_iᵀ, Q_i M_i Q_i
thr = fit.u.quantile(1 - tail, dim=0)
Xi1, Xi1h = xi(fit.u >= thr), xi(held.u >= thr)
a, b = torch.zeros(D, H, device=dev), torch.zeros(H, device=dev)
for s, sl in fit.batches():
    _, F_, G, Z = factors(*fit.at(sl))
    wgt = Xi1[sl] / Z[:, None]
    a += torch.einsum("nkd,nkh,nh->dh", G, F_, wgt)
    b += (F_.square().sum(1) * wgt).sum(0)
wn = torch.nn.functional.normalize(W, dim=0)
ev1, ev1h, nE1, qmq1, chans1 = [], [], [], [], []
tmin = min(GAMMAS) / D
for i in range(0, H, 256):
    sl = slice(i, i + 256)
    Mi, wi, ai = mats(fit, Xi1[:, sl]), W[:, sl].T, a[:, sl].T
    ev, vec = torch.linalg.eigh(Mi.double())
    ev1.append(ev.float())
    for n, m in torch.nonzero(ev >= tmin).tolist():
        chans1.append((ev[n, m].item(), vec[n, :, m].float(), f"n{i + n}", [i + n]))
    ev1h.append(eigs(mats(held, Xi1h[:, sl])))
    Ei = Mi - wi[:, :, None] * ai[:, None, :] - ai[:, :, None] * wi[:, None, :] + b[sl, None, None] * wi[:, :, None] * wi[:, None, :]
    nE1.append(eigs(Ei).abs().max(1).values)
    Q = torch.eye(D, device=dev) - wn[:, sl].T[:, :, None] * wn[:, sl].T[:, None, :]
    qmq1.append(eigs(Q @ Mi @ Q))
ev1, ev1h, nE1, qmq1 = torch.cat(ev1), torch.cat(ev1h), torch.cat(nE1), torch.cat(qmq1)
lam1 = ev1[:, -1]
order = lam1.argsort(descending=True)
print(f"\nsingle-neuron openers (k = 1, on = top {tail:g} of u_i): d·λ₁(M_i) median {D * lam1.median():.1f}, "
      f"90% {D * lam1.quantile(.9):.1f}, max {D * lam1.max():.1f}; d·λ_min min {D * ev1[:, 0].min():.1f}; "
      f"held-out d·λ₁ max {D * ev1h[:, -1].max():.1f}")
print(f"   top 12 by λ₁: " + ", ".join(f"{int(n)} ({D * lam1[n]:.0f})" for n in order[:12]))
summary = {}
for g in GAMMAS:
    t = g / D
    npos, nneg = count(ev1, t)
    hpos, _ = count(ev1h, t)
    q_pos, q_neg = count(qmq1, t)
    qual = (npos + nneg) > 0
    nq = int(qual.sum())
    summary[g] = dict(openers1=nq, openers1_held=int((hpos > 0).sum()), Npos1_max=int(npos.max()), Nneg1_max=int(nneg.max()),
                      thm3_1=int((nE1[qual] < t).sum()), thm3q_1=int((qmq1[qual].abs().max(1).values < t).sum()))
    msg = f"   γ = {g:g}: {nq} neurons open ≥ 1 gate direction (held-out: {int((hpos > 0).sum())})"
    if nq:
        msg += (f" | gate directions per opener N₊ max {int(npos.max())}, N₋ max {int(nneg.max())}, N₊ ≥ 2 for {int((npos >= 2).sum())} | "
                f"Prop 1 budget d·Σ₊λ/γ max {(D * ev1.clamp_min(0).sum(1) / g)[qual].max():.1f} | "
                f"Thm 3 (‖E_i‖ < γ/d) holds for {int((nE1[qual] < t).sum())}/{nq}, Thm 3' (‖QMQ‖ < γ/d) for "
                f"{int((qmq1[qual].abs().max(1).values < t).sum())}/{nq} | N₊ ≤ 1 + N₊(QMQ) violated: "
                f"{int((npos > 1 + q_pos).sum() + (nneg > 1 + q_neg).sum())}")
    print(msg)
print(f"   top 50 openers: ‖E_i‖ / λ₁(M_i) median {(nE1 / lam1)[order[:50]].median():.2f}, ‖Q M Q‖ / λ₁ median "
      f"{(qmq1.abs().max(1).values / lam1)[order[:50]].median():.2f}")

# --- neuron-set openers: greedy search, given sets, random sets, the run's openers; exact E_S in one pass


def set_event(S, Fp=fit):
    c = ((Fp.u[:, S] - mu_u[S]) / sd_u[S]).sum(1)
    cf = ((fit.u[:, S] - mu_u[S]) / sd_u[S]).sum(1)
    return c >= cf.quantile(1 - tail)


def lam_max(S_list):
    return eigs(mats(fit, xi(torch.stack([set_event(S) for S in S_list], 1))))[:, -1]


cand = order[:64].tolist()
greedy = []
for seed in order[:cli.seeds].tolist():
    S = [seed]
    while len(S) < cli.greedy:
        trial = [S + [j] for j in cand if j not in S]
        S = trial[int(lam_max(trial).argmax())]
        greedy.append(list(S))
given = [[int(n) for n in s.split(",")] for s in cli.sets.split(";") if s]
rand_sets = [torch.randperm(H, generator=torch.Generator().manual_seed(100 + i))[:k].tolist() for k in (2, 4) for i in range(16)]
reps, _ = openers(res)
osets = [[int(n) for n in s.split(",")] for s in cli.opener_sets.split(";") if s]
O = res["O"][:, reps].to(dev)
co_fit, co_held = (fit.x - S_.MU) @ S_.IROOT @ O, (held.x - S_.MU) @ S_.IROOT @ O
qo = co_fit.quantile(1 - tail, dim=0)
sets = [("greedy", S, set_event(S), set_event(S, held)) for S in greedy] + \
       [("given", S, set_event(S), set_event(S, held)) for S in given] + \
       [("random", S, set_event(S), set_event(S, held)) for S in rand_sets] + \
       [(f"o{j}", S, co_fit[:, j] >= qo[j], co_held[:, j] >= qo[j]) for j, S in enumerate(osets)]
Xi_s = xi(torch.stack([e for _, _, e, _ in sets], 1))
Ms = mats(fit, Xi_s)
Es = torch.zeros(len(sets), D, D, device=dev)
mass_on = torch.zeros(len(sets), device=dev)
for s, sl in fit.batches():
    _, F_, G, Z = factors(*fit.at(sl))
    for k, (_, S, on, _) in enumerate(sets):
        GS = F_[:, :, S] @ W[:, S].T
        GR = G - GS
        Es[k] += torch.einsum("n,nkd,nke->de", Xi_s[sl, k] / Z, GR, GR)
        mass_on[k] += (GS.square().sum((1, 2)) / Z * on[sl].float()).sum() / on.sum()
evS, evSh = eigs(Ms), eigs(mats(held, xi(torch.stack([e for _, _, _, e in sets], 1))))
evK = eigs(Ms - Es)
nES = evK.new_tensor([torch.linalg.matrix_norm(E.double(), 2).item() for E in Es])
qmqS, overlap, in_span = [], [], []
for k, (_, S, _, _) in enumerate(sets):
    U = torch.linalg.qr(W[:, S])[0]
    Q = torch.eye(D, device=dev) - U @ U.T
    qmqS.append(torch.linalg.eigvalsh((Q @ Ms[k] @ Q).double()).float())
    rest = torch.ones(H, dtype=torch.bool, device=dev)
    rest[S] = False
    overlap.append((U.T @ wn[:, rest]).norm(dim=0).max().item())
    top = torch.linalg.eigh(Ms[k].double())[1][:, -1].float()
    in_span.append((U.T @ top).norm().item())
qmqS = torch.stack(qmqS)
print(f"\nneuron-set openers (on = top {tail:g} of Σ_S standardised u; o*: the run's openers, S its top neurons)")
print(f"   K_S = M - E_S, the part touching span(w_S) (rank ≤ 2k, ≤ k eigenvalues of each sign); overlap: max_j∉S ‖Π_S ŵ_j‖; "
      f"top gate direction in span: ‖Π_S s₁‖")
print(f"   kind   k  set                          d·λ₁  d·λ₂  d·Σ₊λ | ‖E_S‖·d ‖QMQ‖·d | d·λ(K_S) top 2, bottom 2 | overlap  in span | S mass on")
for k, (kind, S, _, _) in enumerate(sets):
    if kind == "random" and k % 4:
        continue
    e, ek = evS[k].flip(0), evK[k].flip(0)
    print(f"   {kind:6s} {len(S)}  {str(S)[:28]:28s} {D * e[0]:5.1f} {D * e[1]:5.1f} {D * evS[k].clamp_min(0).sum():5.1f} | "
          f"{D * nES[k]:7.1f} {D * qmqS[k].abs().max():7.1f} | {D * ek[0]:6.1f} {D * ek[1]:5.1f} {D * ek[-2]:6.1f} {D * ek[-1]:6.1f}   | "
          f"{overlap[k]:.2f}     {in_span[k]:.2f}    | {mass_on[k]:.3f}")
for g in GAMMAS:
    t = g / D
    for kind in ("greedy", "given", "random", "o"):
        ks = [k for k, s in enumerate(sets) if s[0].startswith(kind)]
        if not ks:
            continue
        npos, nneg = count(evS[ks], t)
        hpos, _ = count(evSh[ks], t)
        r = torch.tensor([len(sets[k][1]) for k in ks], device=dev)
        qp, qn = count(qmqS[ks], t)
        summary[g].update({f"{kind}_Npos": npos.tolist(), f"{kind}_thm3": int((nES[ks] < t).sum()), f"{kind}_n": len(ks)})
        print(f"   γ = {g:g}, {kind:6s}: N₊ {npos.tolist()} (held-out {hpos.tolist()}), N₋ max {int(nneg.max())} | k {r.tolist()} | "
              f"Thm 3 holds {int((nES[ks] < t).sum())}/{len(ks)}, Thm 3' {int((qmqS[ks].abs().max(1).values < t).sum())}/{len(ks)} | "
              f"N₊ ≤ r + N₊(QMQ) violated {int((npos > r + qp).sum() + (nneg > r + qn).sum())}")

print(f"\nthe run's attractors (o: top {tail:g} of c(o)); pipeline contrast is g/τ with τ from 8 random pushes")
for j in range(len(osets)):
    k = len(sets) - len(osets) + j
    s = res["S"][:, reps[j]].to(dev)
    ev, vec = torch.linalg.eigh(Ms[k].double())
    print(f"   attractor {j} (S = {osets[j]}): d·sᵀMs {D * (s @ Ms[k] @ s):.1f} (pipeline held-out {res['checks'][j]['contrast']:.1f}); "
          f"|cos(s, top gate direction)| {abs(vec[:, -1].float() @ s):.2f}; d·λ top-4 {np.round(D * ev.flip(0)[:4].cpu().numpy(), 1).tolist()}")

# --- gates: (opener, direction) pairs with d·λ ≥ γ over single neurons and non-random sets; distinct channels (|cos| ≤ 0.8)
chans = list(chans1)
for k, (kind, S, _, _) in enumerate(sets):
    if kind != "random":
        ev, vec = torch.linalg.eigh(Ms[k].double())
        chans += [(ev[m].item(), vec[:, m].float(), kind if kind.startswith("o") else str(S), S) for m in torch.nonzero(ev >= tmin)[:, 0]]
print(f"\ngates: positive gate directions with d·λ ≥ γ, openers = all single neurons + greedy, given and the run's openers")
gate_sets, chan_S = {}, {}
for g in GAMMAS:
    t = g / D
    cs = [c for c in chans if c[0] >= t]
    if not cs:
        print(f"   γ = {g:g}: no gates")
        summary[g].update(pairs=0, channels=0)
        continue
    Sm = torch.stack([c[1] for c in cs], 1)
    lab = connected_components((Sm.T @ Sm).abs().cpu().numpy() > 0.8, directed=False)[1]
    best = [max(np.nonzero(lab == c)[0], key=lambda q: cs[q][0]) for c in range(lab.max() + 1)]
    best.sort(key=lambda q: -cs[q][0])
    Sd = Sm[:, best]
    B = torch.linalg.eigvalsh(((Sd.T @ Sd) ** 2).double())[-1].item()
    B0 = torch.linalg.eigvalsh(((Sd.T @ Sd) ** 2 - 1 / D).double())[-1].item()   # frame bound on trace-zero D (Thm 5″)
    chan_S[g] = Sd.cpu()
    q = torch.cat([torch.einsum("nde,em->ndm", fit.X[i:i + 64].view(-1, D, D), Sd).mul(Sd).sum(1) for i in range(0, fit.N, 64)])
    varq, c = q.var(0), tail * (1 - tail) * t ** 2
    mult = np.bincount(lab)[lab[best]]
    vk = torch.einsum("dm,de,em->m", Sd, V1, Sd)
    summary[g].update(pairs=len(cs), channels=len(best), B=B, B0=B0, thm5=thm5(c / B), thm5_kappa=kappa * B / c,
                      thm5_kappa0=kappa * B0 / c, thm5b=thm5b(c, B)[0])
    gate_sets[g] = [cs[q][3] for q in best]
    print(f"   γ = {g:g}: {len(cs)} (opener, direction) pairs, {len(best)} distinct channels; frame bound B {B:.2f}, trace-zero B₀ {B0:.2f} "
          f"(random: {(1 + math.sqrt(2 * len(best)) / D) ** 2:.2f}); Thm 5: m ≤ "
          f"{thm5(c / B)} (spectral), ≤ {thm5b(c, B)[0]:.1f} (5'), ≤ {kappa * B / c:.1f} (κ), ≤ {kappa * B0 / c:.1f} (κ, B₀); Var(sᵀPs) / p(1-p)(γ/d)²: min {(varq / c).min():.2f}, "
          f"median {(varq / c).median():.2f}")
    for n, q_ in enumerate(best[:8]):
        print(f"      channel {n}: best opener {cs[q_][2]} (d·λ {D * cs[q_][0]:.0f}), opened by {mult[n]} of the openers | "
              f"⟨ssᵀ, V₁(K)⟩ {vk[n]:+.2f}, Var(sᵀPs)/λ₁(K) {varq[n] / lamK[0]:.2f} | E[sᵀPs] {q[:, n].mean():.4f} "
              f"(isotropic {1 / D:.4f})")
    if len(best) > 1:
        print(f"      max |cos| between distinct channels {(Sd.T @ Sd).abs().fill_diagonal_(0).max():.2f}")
    # Thm 7 (linear budget): p·A ≤ d·E[sᵀPs] per gate, so Σ_a p A_a ≤ d·B'·Σ_{i≤m} λ_i(P̄), B' = ‖Σ_a s_a s_aᵀ‖
    pA = tail * D * torch.tensor([cs[q_][0] for q_ in best], device=dev)
    B1 = torch.linalg.eigvalsh(Sd.T @ Sd)[-1].item()
    rhs7 = D * B1 * evP[:len(best)].sum().item()
    avgP = evP.cumsum(0) / torch.arange(1, D + 1, device=dev)
    m7 = int((avgP >= tail * g / (D * B1)).sum())                  # largest m with (1/m)Σ_(i≤m) λ_i(P̄) ≥ pγ/(d B')
    summary[g].update(thm7_lhs=pA.sum().item(), thm7_rhs=rhs7, thm7=m7)
    print(f"      Thm 7: Σ p·A {pA.sum():.1f} ≤ d·B'·Σ_(i≤m) λ_i(P̄) = {rhs7:.1f} (B' {B1:.2f}); per gate p·A / (d·E[sᵀPs]) "
          f"median {(pA / (D * q.mean(0))).median():.2f}; count bound m ≤ {m7} (spectral), ≤ {D * B1 / (tail * g):.1f} (d·B'/(pγ))")

# --- step 5: principal angles between span(w_S) of the openers of distinct channels; diagonal Fisher shares
for g, gs_ in gate_sets.items():
    gs_ = gs_[:40]                                                # strongest 40 channels: the pair loop is quadratic
    if len(gs_) > 1:
        Us = [torch.linalg.qr(W[:, S])[0] for S in gs_]
        cosm = [torch.linalg.svdvals(Us[i].T @ Us[j])[0].item() for i in range(len(Us)) for j in range(i + 1, len(Us))
                if not set(gs_[i]) & set(gs_[j])]
        if cosm:
            print(f"\n[5] γ = {g:g}: largest principal cosine between span(w_S) of disjoint openers of distinct channels: "
                  f"median {np.median(cosm):.2f}, max {max(cosm):.2f} ({len(cosm)} pairs)")
share1 = torch.zeros(H, device=dev)
for s, sl in fit.batches():
    _, F_, _, Z = factors(*fit.at(sl))
    share1 += (F_.square().sum(1) * W.square().sum(0)[None] / Z[:, None]).sum(0)
share1 /= fit.N
srt = share1.sort(descending=True).values
print(f"    single-neuron diagonal Fisher share E[‖f_i‖²‖w_i‖²/Z]: sum {share1.sum():.2f} (1 if no cross terms), top-1 "
      f"{srt[0]:.3f}, top-10 {srt[:10].sum():.3f}, top-100 {srt[:100].sum():.3f}, top-1 / median {srt[0] / srt.median():.0f}")

# --- step 9: window bootstrap of N₊ for the strongest openers
print(f"\n[9] window bootstrap ({cli.boot}) of N₊(γ) for the 6 strongest single-neuron openers and greedy {cli.greedy}-sets")
picks = [("n" + str(int(n)), fit.u[:, n] >= thr[n]) for n in order[:6]] + \
        [(str(S), on) for kind, S, on, _ in sets if kind == "greedy" and len(S) == cli.greedy][:4]
gb = torch.Generator(device=dev).manual_seed(1)
for name, on in picks:
    cnt = torch.zeros(cli.boot, nw, device=dev)
    cnt.scatter_add_(1, torch.randint(nw, (cli.boot, nw), device=dev, generator=gb), torch.ones(cli.boot, nw, device=dev))
    w_r, on_f = cnt[:, fit.win].T, on.float()[:, None]
    Xi_b = w_r * on_f / (w_r * on_f).sum(0) - w_r * (1 - on_f) / (w_r * (1 - on_f)).sum(0)
    evb = eigs(mats(fit, Xi_b))
    print(f"   {name:28s} d·λ₁ {D * evb[:, -1].mean():.1f} ± {D * evb[:, -1].std():.1f}, d·λ₂ {D * evb[:, -2].mean():.1f} ± "
          f"{D * evb[:, -2].std():.1f} | " + ", ".join(f"N₊({g:g}) 5-95% {int(count(evb, g / D)[0].float().quantile(.05))}-"
                                                       f"{int(count(evb, g / D)[0].float().quantile(.95))}" for g in GAMMAS))

out = RUNS / "gate_bound"
out.mkdir(exist_ok=True)
torch.save({"args": vars(cli), "model": S_.args.model, "d": D, "H": H, "kappa": kappa, "lamK": lamK[:256].cpu(),
            "ev1": ev1.cpu(), "ev1h": ev1h.cpu(), "nE1": nE1.cpu(), "qmq1": qmq1.cpu(), "noise": eps.cpu(),
            "sets": [(k_, S) for k_, S, _, _ in sets], "evS": evS.cpu(), "evSh": evSh.cpu(), "nES": nES.cpu(),
            "evK": evK.cpu(), "qmqS": qmqS.cpu(), "share1": share1.cpu(), "summary": summary, "channels": chan_S,
            "pipeline_attractors": len(res["reps"])}, out / f"{path.stem}.pt")
print(f"\nsaved {out / (path.stem + '.pt')}")
