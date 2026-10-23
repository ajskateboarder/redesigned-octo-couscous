"""Is sparsity anticorrelated with PC rank? MLP input z of layer --layer, positions 16-31 of the fitting windows of
downstream_amp.py (OpenWebText, draw 0). Part (1) runs for any --model; (2) and (3) need the transcoders (GPT-2 only).

    (1) per PC: excess kurtosis and |skew| of the coordinate along it, by PC band (Gaussian: 0 and 0)
    (2) per transcoder feature (jacobdunefsky/gpt2small-transcoders, whose input is z with ln_2's affine undone): share of
        its pre-activation variance from each PC band, by the feature's density (share of rows where it is active)
    (3) where each feature's discriminating signal lives: whitened mean shift of its active rows per PC band, minus what
        a random row subset of the same size gives (1/n_active - 1/N per whitened dimension); also per dimension,
        top band vs tail
If rarer features relied more on low-variance PCs, (2) and (3) would shift towards the tail as density falls.
For GPT-2, gpt2_noln, gelu_1l and gelu_2l the last PC (a direction LayerNorm leaves with no variance) is excluded.

Usage: python amp_pc_sparsity.py [--model gpt2|swiglu|gpt2_noln|relu2_12l|...] [--layer 8] [--windows 480]
Output: downstream_amp/checks/amp_pc_sparsity_{model}_L{layer}.log
"""
import argparse

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

import amp_models
from amp_common import BEFORE_TAU, log_to, setup

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=amp_models.MODELS, default="gpt2")
parser.add_argument("--layer", type=int, default=8)
parser.add_argument("--windows", type=int, default=480)
cli = parser.parse_args()
S_ = setup({"model": cli.model, "layer": cli.layer, "windows": cli.windows, "heldout": 8}, stop=BEFORE_TAU)
log_to(f"amp_pc_sparsity_{cli.model}_L{cli.layer}")
M, L, dev, fit, MU, ev, evec, D = S_.M, S_.L, S_.dev, S_.fit, S_.MU, S_.ev, S_.evec, S_.D
torch.set_grad_enabled(False)
E, lam = evec.flip(1).float(), ev.flip(0).float()                # PCs, descending variance
Z = (fit.z().reshape(-1, D) - MU) @ E                       # coordinates along each PC
Zs = (Z - Z.mean(0)) / Z.std(0)
kurt, skew = (Zs ** 4).mean(0) - 3, (Zs ** 3).mean(0).abs()
bands = [(0, 16), (16, 64), (64, 256), (256, D - 1 if cli.model in ("gpt2", "gpt2_noln", "gelu_1l", "gelu_2l") else D)]
print(f"{cli.model} L{L}, {fit.n}×{S_.P_} rows: per-PC coordinate shape by band (median over PCs; Gaussian: kurtosis 0, |skew| 0)")
for a, b in bands:
    print(f"  PCs {a:>3}-{b:<3}: excess kurtosis median {kurt[a:b].median():6.2f} (90th pct {kurt[a:b].quantile(.9):6.2f}) | "
          f"|skew| median {skew[a:b].median():.2f} | variance share {lam[a:b].sum() / lam.sum():.3f}")
if cli.model != "gpt2":
    raise SystemExit
t = {k: v.to(dev) for k, v in load_file(hf_hub_download("jacobdunefsky/gpt2small-transcoders",
                                                        f"gpt2-small-dun-chl-mlp-tc{L}/sae.safetensors")).items()}
ln2 = M.hf.transformer.h[L].ln_2
read = t["W_enc"] / ln2.weight[:, None]                          # pre-activation = read·z + const
pre = ((fit.z().reshape(-1, D) - ln2.bias) / ln2.weight) @ t["W_enc"] + t["b_enc"]
dens = (pre > 0).float().mean(0)
var_pc = (E.T @ read).square() * lam[:, None]                    # (PC, feature): variance of read·z along each PC
share = var_pc / var_pc.sum(0, keepdim=True)
groups = [(0.10, 1.0, "dense   >10%"), (0.01, 0.10, "1-10%"), (0.002, 0.01, "0.2-1%"), (0.0002, 0.002, "rare 0.02-0.2%")]
print(f"\ntranscoder features: share of pre-activation variance from each PC band (median within density group)")
print("  density group      n     " + "  ".join(f"PCs {a}-{b}" for a, b in bands))
for lo, hi, name in groups:
    sel = (dens >= lo) & (dens < hi)
    sh = torch.stack([share[a:b][:, sel].sum(0).median() for a, b in bands])
    print(f"  {name:<16} {int(sel.sum()):>6}   " + "  ".join(f"{v:9.3f}" for v in sh.tolist()))
alive = dens > 0
rho = torch.corrcoef(torch.stack([dens[alive].log(), share[bands[3][0]:bands[3][1]][:, alive].sum(0)]))[0, 1]
print(f"\ncorrelation over {int(alive.sum())} live features: log density vs share from PCs 256+ : {rho:+.2f}")

# (3) where does each feature's discriminating signal live? whitened mean shift of its active rows, per PC band,
#     minus what a random row subset of the same size gives (1/n_A - 1/N per whitened dimension)
Zw = Z / lam.sqrt()                                              # whitened coordinates in the PC basis
N = Zw.shape[0]
act = pre > 0
nA = act.sum(0).float()
ok = nA >= 20
sel_f = torch.nonzero(ok)[:, 0]
excess = []
for f0 in range(0, len(sel_f), 2048):
    idx = sel_f[f0:f0 + 2048]
    a = act[:, idx].float()
    d = (a.T @ Zw) / nA[idx, None] - Zw.mean(0)                  # (features, PCs)
    e = torch.stack([d[:, a_:b_].square().sum(1) - (b_ - a_) * (1 / nA[idx] - 1 / N) for a_, b_ in bands], 1)
    excess.append(e)
excess = torch.cat(excess).clamp_min(0)
sh = excess / excess.sum(1, keepdim=True).clamp_min(1e-12)
dsel = dens[sel_f]
print(f"\ndiscriminating signal (whitened, noise-corrected): share per PC band, median within density group "
      f"({len(sel_f)} features with ≥ 20 active rows)")
print("  density group      n     " + "  ".join(f"PCs {a}-{b}" for a, b in bands) + "   per-dim, 0-16 vs 256+")
for lo, hi, name in groups:
    m = (dsel >= lo) & (dsel < hi)
    if m.sum() == 0:
        continue
    med = sh[m].median(0).values
    (a0, b0), (a3, b3) = bands[0], bands[3]
    per_dim = (excess[m][:, 0] / (b0 - a0)).median() / (excess[m][:, 3] / (b3 - a3)).median().clamp_min(1e-12)
    print(f"  {name:<16} {int(m.sum()):>6}   " + "  ".join(f"{v:9.3f}" for v in med.tolist()) + f"   ×{per_dim:.1f}")
rho = torch.corrcoef(torch.stack([dsel.log(), sh[:, 3]]))[0, 1]
print(f"correlation log density vs signal share from PCs 256+: {rho:+.2f}")
