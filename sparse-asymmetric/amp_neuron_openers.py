"""Is a gate's fitted opener o the direction its neurons actually read? For given neurons of MLP L (e.g. the ones
amp_neurons.py finds carrying an attractor), compare o with each neuron's read direction (its input weights, in the
whitened coordinates o lives in) and use that read direction as the opener itself, with the attractor's channel s.

    |cos|      between o and the neuron's whitened read direction Σ^½ w_in (sign aligned with o)
    as opener  held-out contrast and causal share (as in downstream_amp.py: push the off-rows along the opener to the
               on-rows' mean coordinate) with the neuron's read direction as opener, s projected ⟂ to it
    Jaccard    of the neuron opener's on-set (top --tail of rows) with o's
If o were simply the input that switches these neurons, the read directions would match it and be at least as good openers.

Usage: python amp_neuron_openers.py <run .pt> --attractor J --neurons 251,1608
Output: downstream_amp/checks/amp_neuron_openers_<run>_a<J>.log
"""
import argparse

import torch
from torch.nn import functional as F

from amp_common import from_run, log_to, openers

parser = argparse.ArgumentParser()
parser.add_argument("run")
parser.add_argument("--attractor", type=int, required=True, help="index among the run's reported attractors")
parser.add_argument("--neurons", required=True, help="comma-separated neuron indices of MLP L")
cli = parser.parse_args()
path, res, S_ = from_run(cli.run)
log_to(f"amp_neuron_openers_{path.stem}_a{cli.attractor}")
torch.set_grad_enabled(False)
dev, held, tau_h, coords, sweep, ROOT, tail = S_.dev, S_.held, S_.tau_h, S_.coords, S_.sweep, S_.ROOT, S_.args.tail
P = S_.M.mlp_parts(S_.L)
assert "w_in" in P, "needs a threshold MLP (GELU / ReLU²): neurons with a single read direction"
reps, _ = openers(res)
o, s = res["O"][:, reps[cli.attractor]].to(dev), res["S"][:, reps[cli.attractor]].to(dev)


def contrast_and_causal(o, s):
    c = coords(held, o[:, None])[0]
    on = c >= c.flatten().quantile(1 - tail)
    _, Gb = sweep(held, s[:, None])
    rel = Gb[0] / tau_h
    obs = (rel[on].mean() - rel[~on].mean()).item()
    delta = (c[on].mean() - c[~on].mean()).item()
    _, Gp = sweep(held, s[:, None], shift=delta * (ROOT @ o))
    return obs, ((Gp - Gb)[0][~on] / tau_h[~on]).mean().item() / obs, on


obs, caus, on_o = contrast_and_causal(o, s)
print(f"{S_.args.model} L{S_.L} ({path.name}), attractor {cli.attractor}: fitted o: held-out contrast {obs:.1f}, causal {caus:.2f}")
for k in (int(n) for n in cli.neurons.split(",")):
    u = F.normalize(ROOT @ P["w_in"][:, k].float(), dim=0)
    u = u * torch.sign(u @ o)
    obs_u, caus_u, on_u = contrast_and_causal(u, F.normalize(s - (s @ u) * u, dim=0))
    jac = ((on_u & on_o).sum() / (on_u | on_o).sum()).item()
    print(f"   neuron {k}: |cos(o, read direction)| {(u @ o).item():.2f} | as opener: contrast {obs_u:.1f}, causal {caus_u:.2f} | "
          f"on-set Jaccard with o's {jac:.2f}")
