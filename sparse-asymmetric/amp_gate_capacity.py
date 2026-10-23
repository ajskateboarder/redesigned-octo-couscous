"""Planted-gate capacity: how many gates can a one-hidden-layer MLP of width H store, at input dimension d? The
achievability side of mode_bound.md (§4), which bounds gate counts from above only.

Teacher (ℝ^d → ℝ^d), inputs x ~ N(0, I_d) (so already whitened):
    f*(x) = Q x + α Σ_a σ((o_a·x - t)/T) (s_a·x) u_a
    Q a random orthogonal matrix (isotropic baseline sensitivity), α = --alpha·√d (at 1, one open gate carries as much
    Fisher mass as the whole baseline), o_a, s_a, u_a random unit vectors with s_a ⟂ o_a, t = Φ⁻¹(1 - p) so gate a is open on a share p
    of inputs, T = --sharp. Gate a: opener event E_a = {o_a·x > t}, channel s_a.
Student: f(x) = W₂ φ(W₁x + b₁) + b₂, width H, GELU, no skip (as an MLP in the residual stream: all of its sensitivity
    has to pass through the neurons), trained by MSE on fresh Gaussian batches.
Both are measured as amp_mode_bound.py measures a model, with the readout the identity (C = I): G = J(x),
    P = GᵀG / ‖G‖²_F, M_a = E[P | E_a] - E[P | not E_a], contrast A_a = d·s_aᵀM_a s_a, on --eval fresh inputs.
Gate a is stored if the student's A_a ≥ ½ the teacher's and the top eigenvector of the student's M_a has |cos| ≥ 0.8 with
s_a. Also reported: the noise floor (95% of |A|/d for random row events of prevalence p, on the teacher's rows), how many gates clear absolute thresholds γ/d (teacher and student), the student's function fit (R²)
and Jacobian error, and the budget of Thm 5, Σ_a p(1-p)A_a² / (κ d²) (≤ B for any model).

Usage: python amp_gate_capacity.py [--d 64] [--m 1,2,4,...] [--H 16,32,...] [--p 0.1] [--seed 0] [--tag main]
Output: downstream_amp/checks/amp_gate_capacity_<tag>.log, downstream_amp/mode_bound/capacity_<tag>.pt
"""
import argparse
import math
import time

import numpy as np
import torch
from torch.nn import functional as F

from amp_common import RUNS, log_to

parser = argparse.ArgumentParser()
parser.add_argument("--d", default="64", help="input / output dimensions (comma list)")
parser.add_argument("--m", default="1,2,4,8,16,32,64,128,256", help="planted gates (comma list)")
parser.add_argument("--H", default="16,32,64,128,256,512,1024,2048", help="student widths (comma list)")
parser.add_argument("--p", default="0.1", help="gate prevalences (comma list)")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--alpha", type=float, default=1.0, help="gate strength in units of √d")
parser.add_argument("--sharp", type=float, default=0.25, help="opener sigmoid temperature T")
parser.add_argument("--steps", type=int, default=6000)
parser.add_argument("--batch", type=int, default=4096)
parser.add_argument("--eval", type=int, default=16384)
parser.add_argument("--gamma", default="0.05,0.1,0.2", help="absolute contrast thresholds γ/d")
parser.add_argument("--tag", default="main")
cli = parser.parse_args()
log_to(f"amp_gate_capacity_{cli.tag}")
dev = torch.device("cuda")
torch.set_float32_matmul_precision("high")
lists = lambda s, t: [t(v) for v in s.split(",")]
GAMMAS = lists(cli.gamma, float)


class Teacher:
    def __init__(self, d, m, p, g):
        self.d, self.p = d, p
        self.Q = torch.linalg.qr(torch.randn(d, d, device=dev, generator=g))[0]
        O, S, U = (F.normalize(torch.randn(d, m, device=dev, generator=g), dim=0) for _ in range(3))
        self.O, self.S, self.U = O, F.normalize(S - (S * O).sum(0) * O, dim=0), U
        self.alpha, self.t = cli.alpha * math.sqrt(d), float(torch.distributions.Normal(0, 1).icdf(torch.tensor(1 - p)))

    def __call__(self, x):
        gate = torch.sigmoid((x @ self.O - self.t) / cli.sharp)
        return x @ self.Q.T + self.alpha * (gate * (x @ self.S)) @ self.U.T

    def jac(self, x):
        z = (x @ self.O - self.t) / cli.sharp
        sg = torch.sigmoid(z)
        dsg = sg * (1 - sg) / cli.sharp * (x @ self.S)
        return self.Q + self.alpha * (torch.einsum("dm,nm,em->nde", self.U, sg, self.S) +
                                      torch.einsum("dm,nm,em->nde", self.U, dsg, self.O))


def train(teacher, H, g):
    d = teacher.d
    l1, l2 = torch.nn.Linear(d, H, device=dev), torch.nn.Linear(H, d, device=dev)
    opt = torch.optim.Adam([*l1.parameters(), *l2.parameters()], lr=3e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=cli.steps, pct_start=0.05)
    for _ in range(cli.steps):
        x = torch.randn(cli.batch, d, device=dev, generator=g)
        loss = (l2(F.gelu(l1(x))) - teacher(x)).square().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    W1, b1, W2 = l1.weight.detach(), l1.bias.detach(), l2.weight.detach()
    slope = lambda u: 0.5 * (1 + torch.erf(u / math.sqrt(2))) + u * torch.exp(-u ** 2 / 2) / math.sqrt(2 * math.pi)
    f = lambda x: l2(F.gelu(l1(x))).detach()
    jac = lambda x: torch.einsum("dh,nh,he->nde", W2, slope(x @ W1.T + b1), W1)
    return f, jac


def fingerprints(J):
    G = J.flatten(1)
    return ((J.transpose(1, 2) @ J) / G.square().sum(1)[:, None, None]).flatten(1)


def gates(X, on, S, d):
    on = on.float()
    Xi = on / on.sum(0) - (1 - on) / (1 - on).sum(0)
    Ms = (Xi.T @ X).view(-1, d, d)
    A = d * torch.einsum("dm,mde,em->m", S, Ms, S)
    vec = torch.linalg.eigh(Ms.double())[1][:, :, -1].float()
    kappa = (X.square().sum(1).mean() - X.mean(0).square().sum()).item()     # E‖P - P̄‖²_F
    return A, (vec * S.T).sum(1).abs(), kappa


def evaluate(teacher, f, jac, g):
    d = teacher.d
    x = torch.randn(cli.eval, d, device=dev, generator=g)
    on = x @ teacher.O > teacher.t
    Xt, Xs = [], []
    for c in x.split(2048):
        Xt.append(fingerprints(teacher.jac(c)))
        Xs.append(fingerprints(jac(c)))
    At, cos_t, kt = gates(torch.cat(Xt), on, teacher.S, d)
    As, cos_s, ks = gates(torch.cat(Xs), on, teacher.S, d)
    y, yh = teacher(x[:4096]), f(x[:4096])
    r2 = 1 - (y - yh).square().sum() / (y - y.mean(0)).square().sum()
    Jt, Js = teacher.jac(x[:2048]), jac(x[:2048])
    jerr = (Js - Jt).square().sum() / Jt.square().sum()
    pq = teacher.p * (1 - teacher.p)
    rand = torch.rand(cli.eval, 64, device=dev, generator=g) < teacher.p
    null = gates(torch.cat(Xt), rand, F.normalize(torch.randn(d, 64, device=dev, generator=g), dim=0), d)[0]
    return dict(null=(null.abs().quantile(0.95) / d).item(), At=At.cpu(), As=As.cpu(), cos_t=cos_t.cpu(), cos_s=cos_s.cpu(), kappa_t=kt, kappa_s=ks, r2=r2.item(),
                jerr=jerr.item(), budget_t=(pq * At.square().sum() / (kt * d * d)).item(),
                budget_s=(pq * As.clamp_min(0).square().sum() / (ks * d * d)).item())


results = []
print(f"planted gates: f*(x) = Qx + {cli.alpha:g}·√d Σ_a σ((o_a·x - t)/{cli.sharp:g})(s_a·x)u_a; student GELU MLP, {cli.steps} steps × "
      f"{cli.batch}; stored = student contrast ≥ ½ teacher's and |cos(top mode, s_a)| ≥ 0.8; counts at γ/d = {GAMMAS}")
for d in lists(cli.d, int):
    for p in lists(cli.p, float):
        print(f"\nd = {d}, p = {p:g}")
        print(f"  {'m':>4s} {'H':>5s} | {'R²':>6s} {'J err':>6s} | teacher A/d med {'':4s} | stored {'':3s} | "
              + " ".join(f"N(γ/d≥{gm:g}) t/s".rjust(17) for gm in GAMMAS) + " | budget t / s | null")
        for m in lists(cli.m, int):
            g = torch.Generator(device=dev).manual_seed(1000 * cli.seed + m)
            teacher = Teacher(d, m, p, g)
            for H in lists(cli.H, int):
                t0 = time.time()
                f, jac = train(teacher, H, torch.Generator(device=dev).manual_seed(cli.seed * 7919 + H))
                r = evaluate(teacher, f, jac, torch.Generator(device=dev).manual_seed(12345))
                stored = ((r["As"] >= 0.5 * r["At"]) & (r["cos_s"] >= 0.8)).sum().item()
                r.update(d=d, p=p, m=m, H=H, stored=stored)
                results.append(r)
                counts = " ".join(f"{int((r['At'] >= gm * d).sum()):>7d} / {int((r['As'] >= gm * d).sum()):<5d}" for gm in GAMMAS)
                print(f"  {m:4d} {H:5d} | {r['r2']:6.3f} {r['jerr']:6.3f} | {r['At'].median() / d:6.3f} (cos {r['cos_t'].median():.2f}) | "
                      f"{stored:4d} {stored / m:5.2f} | {counts} | {r['budget_t']:.2f} / {r['budget_s']:.2f} | {r['null']:.3f} ({time.time() - t0:.0f} s)",
                      flush=True)

print("\ncapacity m*(H): largest m with ≥ 90% of planted gates stored; most gates stored, and most student gates above γ, over m")
for d in lists(cli.d, int):
    for p in lists(cli.p, float):
        cells = []
        for H in lists(cli.H, int):
            ok = [r for r in results if r["d"] == d and r["p"] == p and r["H"] == H and r["stored"] >= 0.9 * r["m"]]
            best = max((r["stored"] for r in results if r["d"] == d and r["p"] == p and r["H"] == H), default=0)
            n1 = max((int((r["As"] >= GAMMAS[1] * d).sum()) for r in results if r["d"] == d and r["p"] == p and r["H"] == H), default=0)
            cells.append(f"H {H}: m* {max((r['m'] for r in ok), default=0)}, max stored {best}, max N(γ/d ≥ {GAMMAS[1]:g}) {n1}")
        print(f"  d = {d}, p = {p:g}: " + "; ".join(cells))
out = RUNS / "mode_bound"
out.mkdir(exist_ok=True)
torch.save({"args": vars(cli), "results": results}, out / f"capacity_{cli.tag}.pt")
print(f"saved {out / f'capacity_{cli.tag}.pt'}")
