"""Transcoder features as openers, and how a run's attractor openers relate to them (GPT-2 small only: the transcoders,
jacobdunefsky/gpt2small-transcoders, were trained on it; their input is the MLP input z with ln_2's affine undone).
Kept as a record of the pilot; the transcoder and the attractors are not a like-for-like basis (the transcoder is defined
by reconstruction and sparsity, the attractors by their effect on predictions), so feature counts here inherit the
transcoder's splitting and absorption.

    pilot     (--nfeat N > 0) N random features with density 1-10% on the fitting rows as fixed openers, "on" where the
              feature is active; for each, s = the top eigenvector of the contrast ⟂ o (15 shifted power steps on the
              fitting rows), then the held-out contrast. Null: random whitened directions whose on-set is their top tail
              with the same density as the matched feature
    matching  per attractor of the run, on held-out rows: best single-feature Jaccard with its on-set (features with
              ≥ 10 active rows); the 5 features most contained in it (share of the feature's active rows inside, ≥ 20
              active rows) and how much of the on-set they cover together; how many contained features (≥ 50% inside,
              at most twice the on-set's size) it takes, greedily, to cover half of it ("-" if they never do); |corr|
              over held-out rows between the opener's coordinate and the most contained feature's pre-activation

Usage: python amp_transcoder.py <run .pt in downstream_amp/ or a path> [--nfeat 64]
Output: downstream_amp/checks/amp_transcoder_<run>.log
"""
import argparse

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch.nn import functional as F

from amp_common import BEFORE_TAU, FIT_LOOP, from_run, log_to, openers

parser = argparse.ArgumentParser()
parser.add_argument("run")
parser.add_argument("--nfeat", type=int, default=64, help="features to test as openers (0: matching only)")
cli = parser.parse_args()
path, res, S_ = from_run(cli.run, stop=FIT_LOOP if cli.nfeat else BEFORE_TAU)
log_to(f"amp_transcoder_{path.stem}")
assert S_.args.model == "gpt2", "the transcoders were trained on the original GPT-2 small"
M, L, D, dev, P_, FIRST, tail = S_.M, S_.L, S_.D, S_.dev, S_.P_, S_.FIRST, S_.args.tail
fit, held, sweep, coords = S_.fit, S_.held, S_.sweep, S_.coords
tau_f, tau_h = getattr(S_, "tau_f", None), getattr(S_, "tau_h", None)
ROOT, IROOT, tok = S_.ROOT, S_.IROOT, M.tokenizer
perp = lambda S, O: F.normalize(S - (S * O).sum(0, keepdim=True) * O, dim=0)
torch.set_grad_enabled(False)

t = {k: v.to(dev) for k, v in load_file(hf_hub_download("jacobdunefsky/gpt2small-transcoders",
                                                        f"gpt2-small-dun-chl-mlp-tc{L}/sae.safetensors")).items()}
ln2 = M.hf.transformer.h[L].ln_2
read = t["W_enc"] / ln2.weight[:, None]                          # pre-activation = read·z + const
pre = lambda R, idx: ((R.z() - ln2.bias) / ln2.weight) @ t["W_enc"][:, idx] + t["b_enc"][idx]
n_feat = t["W_enc"].shape[1]
print(f"gpt2 L{L} ({path.name}): transcoder with {n_feat} features")

if cli.nfeat:
    dens = torch.zeros(n_feat, device=dev)
    for s0 in range(0, fit.n, 64):
        dens += ((((fit.z()[s0:s0 + 64] - ln2.bias) / ln2.weight) @ t["W_enc"] + t["b_enc"]) > 0).float().sum((0, 1))
    dens /= fit.n * P_
    gen = torch.Generator(device=dev).manual_seed(0)
    pool = torch.nonzero((dens >= 0.01) & (dens <= 0.10))[:, 0]
    feats = pool[torch.randperm(len(pool), device=dev, generator=gen)[:cli.nfeat]]
    print(f"pilot: {len(pool)} features with density 1-10% on the fitting rows; testing a random {len(feats)}")
    weights = lambda on, tau: (on.float() / on.sum((0, 1)) - (~on).float() / (~on).sum((0, 1))) / tau[..., None]

    def top_channel(O, on_fit, iters=15):
        Wt = weights(on_fit, tau_f)
        S = perp(torch.randn(D, O.shape[1], device=dev, generator=gen), O)
        sigma = None
        for _ in range(iters):
            grad, _ = sweep(fit, S, Wt)
            sigma = grad.norm(dim=0) if sigma is None else sigma
            S = perp(F.normalize(grad + sigma * S, dim=0), O)
        return S

    def heldout(S, on_h):
        rel = sweep(held, S)[1] / tau_h
        return torch.stack([r[o].mean() - r[~o].mean() for r, o in zip(rel, on_h.permute(2, 0, 1))])

    O_tc = F.normalize(ROOT @ read[:, feats], dim=0)
    A_tc = heldout(top_channel(O_tc, pre(fit, feats) > 0), pre(held, feats) > 0)
    O_r = F.normalize(torch.randn(D, len(feats), device=dev, generator=gen), dim=0)
    c_fit, c_h = coords(fit, O_r).permute(1, 2, 0), coords(held, O_r).permute(1, 2, 0)
    thr = torch.stack([c_fit[..., k].flatten().quantile(1 - dens[f]) for k, f in enumerate(feats.tolist())])
    A_r = heldout(top_channel(O_r, c_fit >= thr), c_h >= thr)
    pct = lambda v: " / ".join(f"{x:.3g}" for x in v.quantile(torch.tensor([.1, .5, .9], device=dev)).tolist())
    print(f"  held-out contrast (10th / 50th / 90th pct): transcoder openers {pct(A_tc)} | random, density-matched {pct(A_r)}")
    print(f"  share of transcoder openers above the null's 99th pct ({A_r.quantile(.99):.3g}): "
          f"{(A_tc > A_r.quantile(.99)).float().mean():.2f}; max {A_tc.max():.3g}")
    ctx = lambda row: (tok.decode(held.ids[row // P_, max(0, row % P_ + FIRST - 5):row % P_ + FIRST]) + "[" +
                       tok.decode(held.ids[row // P_, row % P_ + FIRST:row % P_ + FIRST + 1]) + "]").replace("\n", "⏎")
    for k in A_tc.argsort(descending=True)[:5].tolist():
        f = int(feats[k])
        top = pre(held, feats[k:k + 1])[..., 0].flatten().argsort(descending=True)[:5].tolist()
        print(f"  feature {f} (density {dens[f]:.3f}): held-out contrast {A_tc[k]:.3g} | top: {' | '.join(ctx(x) for x in top)}")

reps, rec = openers(res)
O_a = res["O"][:, reps].to(dev)
c_a = coords(held, O_a).flatten(1)
on_a = c_a >= c_a.quantile(1 - tail, dim=1, keepdim=True)        # (attractors, rows)
act = torch.cat([pre(held, slice(f0, f0 + 4096)) > 0 for f0 in range(0, n_feat, 4096)], -1).reshape(-1, n_feat)
n_act = act.sum(0).float()
print(f"matching on {act.shape[0]} held-out rows (on-set = top {tail:g} of rows)")
for j in range(len(reps)):
    oj = on_a[j]
    inside = (act & oj[:, None]).sum(0).float()
    jac = torch.where(n_act >= 10, inside / (n_act + oj.sum() - inside).clamp_min(1), torch.zeros_like(n_act))
    contain = torch.where(n_act >= 20, inside / n_act.clamp_min(1), torch.zeros_like(n_act))
    best = contain.topk(5).indices
    cover5 = (act[:, best].any(-1) & oj).sum() / oj.sum()
    cand = torch.where((n_act >= 10) & (n_act <= 2 * oj.sum()), inside / n_act.clamp_min(1), torch.zeros_like(n_act))
    covered, n_cover = torch.zeros_like(oj), None
    for i, f in enumerate(cand.argsort(descending=True).tolist()):
        if cand[f] < 0.5:
            break
        covered |= act[:, f] & oj
        if covered.sum() >= 0.5 * oj.sum():
            n_cover = i + 1
            break
    u, v = c_a[j], pre(held, best[:1]).flatten()
    corr = (((u - u.mean()) * (v - v.mean())).mean() / (u.std() * v.std())).abs()
    ex = ", ".join(f"{int(f)} ({contain[f]:.2f} of {int(n_act[f])})" for f in best.tolist())
    print(f"  attractor {j} ({'recurring' if rec[j] else 'one-off'}): best Jaccard {jac.max():.2f} | most contained: {ex} | "
          f"those 5 cover {cover5:.2f} | to cover half: {n_cover if n_cover else '-'} | |corr| with best {corr:.2f}")
