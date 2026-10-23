"""Anatomy of one gate: is its "open" condition a single direction, or something no single direction expresses?

The gate is attractor J of a run (opener o, channel s); its "open" rows are the top --tail of rows by the channel's gain
g(s)/τ (fitting rows for training the predictors below, held-out rows for evaluating them). Predictors of "open", scored
by held-out AUC (0.5 = chance, 1 = perfect):
    o              the run's fitted opener coordinate c(o)
    s, |s|         the channel's own coordinate c(s) and its magnitude: a self-gate (the MLP passing s according to how
                   much s is present, as in a noise gate / threshold amplifier on one direction) would be predicted by these
    best direction the best single linear direction of the whitened MLP input (logistic regression on the fitting rows);
                   any single-direction gate, opener-style or one neuron's read vector, is at most this good
    neuron k       one of the given neurons alone (its GELU output; a threshold on its read direction)
    neurons        the given neurons together (logistic regression on their GELU outputs and pairwise products)
Then P(open) for o on/off × s above/below its median, and, for the first two neurons: how often the gate is open in each on/off combination (a neuron counts as on where
GELU'(pre-activation) > 0.5, i.e. past its threshold), how the open rows split among the combinations, and example contexts.

Usage: python amp_gate_anatomy.py <run .pt> --attractor J --neurons 251,1608,1942,690
Output: downstream_amp/checks/amp_gate_anatomy_<run>_a<J>.log
"""
import argparse
import math

import torch
from torch.nn import functional as F

from amp_common import from_run, log_to, openers

parser = argparse.ArgumentParser()
parser.add_argument("run")
parser.add_argument("--attractor", type=int, required=True)
parser.add_argument("--neurons", required=True, help="comma-separated neuron indices of MLP L (first two are tabulated)")
parser.add_argument("--examples", type=int, default=4)
cli = parser.parse_args()
path, res, S_ = from_run(cli.run)
log_to(f"amp_gate_anatomy_{path.stem}_a{cli.attractor}")
torch.set_grad_enabled(False)
dev, fit, held, tau_f, tau_h, sweep, coords = S_.dev, S_.fit, S_.held, S_.tau_f, S_.tau_h, S_.sweep, S_.coords
MU, IROOT, P_, FIRST, tail, tok = S_.MU, S_.IROOT, S_.P_, S_.FIRST, S_.args.tail, S_.tok
P = S_.M.mlp_parts(S_.L)
assert "pre" in P, "needs a threshold MLP (neurons with a single read direction)"
neurons = [int(n) for n in cli.neurons.split(",")]
reps, _ = openers(res)
o, s = res["O"][:, reps[cli.attractor]].to(dev), res["S"][:, reps[cli.attractor]].to(dev)


def rows(R):
    rel = (sweep(R, s[:, None])[1][0] / tau_f if R is fit else sweep(R, s[:, None])[1][0] / tau_h).flatten()
    z = R.z().reshape(-1, R.z().shape[-1])
    u = P["pre"](R.z())[..., neurons].reshape(-1, len(neurons))
    return dict(rel=rel, z=z, white=(z - MU) @ IROOT, c=coords(R, o[:, None])[0].flatten(),
                cs=coords(R, s[:, None])[0].flatten(), pre=u, act=F.gelu(u))


Fr, Hr = rows(fit), rows(held)
y_fit = Fr["rel"] >= Fr["rel"].quantile(1 - tail)
y = Hr["rel"] >= Hr["rel"].quantile(1 - tail)


def auc(score, lab):
    r = score.argsort().argsort().float() + 1
    n1, n0 = lab.sum().float(), (~lab).sum().float()
    return ((r[lab].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)).item()


def logistic(Xf, yf, Xh, steps=600, wd=1e-3):
    mu, sd = Xf.mean(0), Xf.std(0).clamp_min(1e-6)
    Xf, Xh = (Xf - mu) / sd, (Xh - mu) / sd
    w = torch.zeros(Xf.shape[1], device=dev, requires_grad=True)
    b = torch.zeros((), device=dev, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=0.05)
    with torch.enable_grad():
        for _ in range(steps):
            loss = F.binary_cross_entropy_with_logits(Xf @ w + b, yf.float()) + wd * w.square().sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return (Xh @ w + b).detach(), F.normalize(w.detach(), dim=0)


def pairwise(A):
    i, j = torch.triu_indices(A.shape[1], A.shape[1], 1)
    return torch.cat([A, A[:, i] * A[:, j]], 1)


best_score, best_w = logistic(Fr["white"], y_fit, Hr["white"])
nn_score, _ = logistic(pairwise(Fr["act"]), y_fit, pairwise(Hr["act"]))
print(f"{S_.args.model} L{S_.L} ({path.name}), attractor {cli.attractor}: gate open = top {tail:g} of rows by g(s)/τ; "
      f"{fit.n * P_} fitting rows, {held.n * P_} held-out rows")
print(f"  held-out AUC for 'open': fitted opener o {auc(Hr['c'], y):.3f} | best single direction {auc(best_score, y):.3f} "
      f"(|cos| with o {abs((best_w @ o).item()):.2f}, with s {abs((best_w @ s).item()):.2f}) | neurons {neurons} together "
      f"{auc(nn_score, y):.3f}")
print(f"  the channel's own coordinate: s {max(auc(Hr['cs'], y), auc(-Hr['cs'], y)):.3f}, |s| {auc(Hr['cs'].abs(), y):.3f} "
      f"(best of either sign)")
print("  each neuron alone: " + ", ".join(f"{n} {auc(Hr['act'][:, k], y):.3f}" for k, n in enumerate(neurons)))

slope = lambda u: 0.5 * (1 + torch.erf(u / math.sqrt(2))) + u * torch.exp(-u ** 2 / 2) / math.sqrt(2 * math.pi)
on = slope(Hr["pre"][:, :2]) > 0.5
on_o = Hr["c"] >= Hr["c"].quantile(1 - tail)
n1, n2 = neurons[:2]
print(f"\n  neurons {n1}, {n2} (on = past threshold); P(open) overall {y.float().mean():.2f}")
ids, P_rows = held.ids, P_
ctx = lambda r: (tok.decode(ids[r // P_rows, max(0, r % P_rows + FIRST - 6):r % P_rows + FIRST]) + "[" +
                 tok.decode(ids[r // P_rows, r % P_rows + FIRST:r % P_rows + FIRST + 1]) + "]").replace("\n", "⏎")
for a in (False, True):
    for b in (False, True):
        m = (on[:, 0] == a) & (on[:, 1] == b)
        if m.sum() == 0:
            continue
        print(f"    {n1} {'on ' if a else 'off'}, {n2} {'on ' if b else 'off'}: {int(m.sum()):>5} rows | P(open) {y[m].float().mean():.2f} | "
              f"share of all open rows {(y & m).sum() / y.sum():.2f} | o on among these open rows {on_o[y & m].float().mean():.2f}")
        good = torch.nonzero(y & m)[:, 0]
        if len(good):
            pick = good[Hr["rel"][good].argsort(descending=True)][:cli.examples * 4:4].tolist()
            print("        open, e.g.: " + " | ".join(ctx(r) for r in pick))
print(f"  open rows where o is on {on_o[y].float().mean():.2f}; o's on-rows that are open {y[on_o].float().mean():.2f}")
sign = 1.0 if auc(Hr["cs"], y) >= 0.5 else -1.0
s_hi = sign * Hr["cs"] >= (sign * Hr["cs"]).median()
print(f"  o × s (s high = above its median, on the side that favours open): P(open)")
for a_ in (False, True):
    print("    o " + ("on " if a_ else "off") + ": " + " | ".join(
        f"s {'high' if b_ else 'low '} {y[(on_o == a_) & (s_hi == b_)].float().mean():.2f} ({int(((on_o == a_) & (s_hi == b_)).sum())} rows)"
        for b_ in (False, True)))
