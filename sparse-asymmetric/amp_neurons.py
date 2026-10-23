"""Neuron-level breakdown of a run's attractors in MLP L: which neurons carry each held-out contrast A(o, s), and is the
gating inside this MLP at all?

    a_k(ρ)    change of neuron k's activation when the MLP input of row ρ is pushed along s (forward-mode derivative)
    split     the gain is quadratic in the MLP's output change Σ_k a_k W_out[k], so with the contrast's row weights
              w_ρ = onoff/τ the contrast splits exactly: A = Σ_k A_k, A_k = ½ Σ_ρ a_k ∂(Σ w g)/∂a_k (shared credit is
              split evenly between the neurons involved). Reported: neurons carrying 50% / 80% of the positive part
    frozen    the contrast left when every neuron's response a_k is replaced by its mean over rows (per position), the
              later blocks still running on each row's own state: ≈ A means the gating happens after this MLP, ≈ 0
              means it is the MLP's neurons that respond differently on on- and off-rows
    switch    for the neurons carrying half of A: share of off- / on-rows where |a_k| < 10% of its typical on-row size,
              and the on/off ratio of typical |a_k| (median, or mean where the median is 0, as for ReLU² neurons that are
              off on most rows; per neuron capped at ×100); for threshold MLPs (GELU, ReLU²) also the share of rows where the
              neuron is active (pre-activation > 0), off → on, and for GELU the slope GELU'(pre-activation), off → on
    gated     a_k = silu(l·x)(r·p) + silu'(l·x)(r·x)(l·p), p the push (for a plain bilinear MLP (l·x)(r·p) + (r·x)(l·p)):
              which term carries it, and how o changes the gate factor silu(l·x) (l·x) and the value factor r·x

Usage: python amp_neurons.py <run .pt in downstream_amp/ or a path>
Output: downstream_amp/checks/amp_neurons_<run>.log
"""
import sys

import torch
from torch.func import grad, jvp

from amp_common import from_run, log_to, openers

path, res, S_ = from_run(sys.argv[1])
log_to(f"amp_neurons_{path.stem}")
M, L, FIRST, dev = S_.M, S_.L, S_.FIRST, S_.dev
held, tau_h, coords, onoff, ROOT, rows_of, sweep = S_.held, S_.tau_h, S_.coords, S_.onoff, S_.ROOT, S_.rows_of, S_.sweep
model, tail = S_.args.model, S_.args.tail
reps, _ = openers(res)
O, S = res["O"][:, reps].to(dev), res["S"][:, reps].to(dev)

P = M.mlp_parts(L)
hidden, W_out, out_bias, after, kind = P["hidden"], P["W_out"], P["bias"], P["after"], P["kind"]


def gain_rows(C, p):
    mu = (p * C).sum(-1, keepdim=True)
    return 0.5 * ((p * (C - mu).square()).sum(-1) + (1 - p.sum(-1)).clamp_min(0) * mu[..., 0].square())


torch.set_grad_enabled(False)
n_hidden = W_out.shape[0]
print(f"{model} L{L} ({path.name}): {n_hidden} neurons, {len(reps)} attractors")
batch = 4 if S_.D > 1000 else 8
for j in range(len(reps)):
    o, s = O[:, j], S[:, j]
    push = ROOT @ s
    c = coords(held, o[:, None])[0]
    on = c >= c.flatten().quantile(1 - tail)
    W = (onoff(c[None]) / tau_h)[0]                              # (n, P_)
    A_k = torch.zeros(n_hidden, device=dev)
    a_all = torch.zeros(held.n, held.x.shape[1], n_hidden, device=dev)

    def gains(a_, sl):
        # per-row gains of MLP-output change a_ @ W_out (rows sl), through the row's own later blocks and readout
        x, r, ctx = held.x[sl], held.r[sl], rows_of(held.ctx, sl)
        h0 = r + hidden(x) @ W_out.to(x.dtype) + out_bias
        C = jvp(lambda eta: M.head(after(h0 + eta, ctx)[:, FIRST:], held.idx[sl]), (torch.zeros_like(h0),), (a_ @ W_out,))[1]
        return gain_rows(C, held.p[sl])

    for b0 in range(0, held.n, batch):
        sl = slice(b0, b0 + batch)
        x = held.x[sl]
        a = jvp(hidden, (x,), (push.expand_as(x),))[1]
        with torch.enable_grad():
            ga = grad(lambda a_: (W[sl] * gains(a_, sl)).sum())(a)
        A_k += 0.5 * (a * ga).sum((0, 1))
        a_all[sl] = a
    a_rows = a_all[:, FIRST:]
    # gating that survives when every neuron's response to the push is frozen at its all-row mean (per position)
    a_bar = a_all.mean(0, keepdim=True)
    frozen = sum((W[b0:b0 + batch] * gains(a_bar.expand(min(batch, held.n - b0), -1, -1), slice(b0, b0 + batch))).sum()
                 for b0 in range(0, held.n, batch)).item()
    rel = sweep(held, s[:, None])[1][0] / tau_h
    obs = (rel[on].mean() - rel[~on].mean()).item()
    tot = A_k.sum().item()
    pos = A_k.clamp_min(0)
    srt = pos.sort(descending=True).values
    cum = srt.cumsum(0) / srt.sum()
    n50, n80 = int((cum < 0.5).sum()) + 1, int((cum < 0.8).sum()) + 1
    pr = (pos.sum() ** 2 / pos.square().sum()).item()
    top = pos.argsort(descending=True)[:n50]
    wt = pos[top] / pos[top].sum()
    A_abs = a_rows[..., top].abs()                               # (n, P_, n50)
    # typical size per neuron: the median, or the mean where the median is 0 (neurons silent on most rows, as ReLU² ones
    # are: their response is exactly 0 whenever they are off)
    typ = lambda v: torch.where(v.median(0).values > 0, v.median(0).values, v.mean(0))
    typ_on, typ_off = typ(A_abs[on]), typ(A_abs[~on])
    shut = A_abs < 0.1 * typ_on                                  # per neuron: under 10% of its typical on-row size
    shut_off = (shut[~on].float().mean(0) * wt).sum().item()
    shut_on = (shut[on].float().mean(0) * wt).sum().item()
    ratio = ((typ_on / torch.maximum(typ_off, 0.01 * typ_on).clamp_min(1e-12)) * wt).sum().item()   # capped at ×100
    line = (f"  attractor {j}: contrast {tot:.1f} (summed over neurons; measured directly {obs:.1f}) | frozen neurons "
            f"{frozen:.1f} ({frozen / tot:.2f} of it) | positive part carried 50% by {n50} neurons, 80% by {n80} "
            f"(participation ratio {pr:.0f}) | those {n50}: shut on off-rows {shut_off:.2f}, on on-rows {shut_on:.2f}, "
            f"median |a| on/off ×{ratio:.1f}")
    if "pre" in P:
        live = (P["pre"](held.x[:, FIRST:])[..., top] > 0).float()
        line += (f" | active (pre > 0) on off-rows {(live[~on].mean(0) * wt).sum():.2f}, on on-rows "
                 f"{(live[on].mean(0) * wt).sum():.2f}")
    if kind == "gelu":
        slope = a_rows[..., top] / (push @ P["w_in"][:, top])    # GELU'(pre-activation) of each top neuron
        line += f" | GELU slope median off {(slope[~on].median(0).values * wt).sum():.2f} → on {(slope[on].median(0).values * wt).sum():.2f}"
    print(line, flush=True)
    if kind not in ("gated", "bilinear"):
        print(f"     top neurons {top[:5].tolist()}", flush=True)
    if kind in ("gated", "bilinear"):
        # which term carries the push, and which factor o changes
        t = top[:50]
        xs = held.x[:, FIRST:]
        gl = P["left"](xs)[..., t]
        rv = P["right"](xs)[..., t]
        lp, rp = P["W_left"][t] @ push, P["W_right"][t] @ push
        if kind == "gated":
            sig = torch.sigmoid(gl)
            t1 = (gl * sig * rp).abs()
            t2 = (sig * (1 + gl * (1 - sig)) * rv * lp).abs()
            gate = (gl * sig).abs()
        else:
            t1, t2, gate = (gl * rp).abs(), (rv * lp).abs(), gl.abs()
        wt_ = pos[t] / pos[t].sum()
        share1 = ((t1 / (t1 + t2))[on].median(0).values * wt_).sum().item()
        g_ratio = ((gate[on].median(0).values / gate[~on].median(0).values) * wt_).sum().item()
        v_ratio = ((rv.abs()[on].median(0).values / rv.abs()[~on].median(0).values) * wt_).sum().item()
        print(f"     top neurons {t[:5].tolist()}: push enters via r (term silu(l·x)(r·p)) {share1:.2f}, via l {1 - share1:.2f} | "
              f"on/off: |silu(l·x)| ×{g_ratio:.1f}, |r·x| ×{v_ratio:.1f}", flush=True)
