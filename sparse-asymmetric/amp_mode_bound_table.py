"""Table of amp_mode_bound.py results across models (mode_bound.md steps 7-8): for each run in downstream_amp/mode_bound/,
d_mlp, activation, optimizer, seed, how much of the pretrained MLP's loss reduction the retrained MLP recovers, the
pipeline's attractor count (downstream_amp.py), the kernel (κ, λ₁(K)), and per γ: single-neuron openers with a mode,
largest N₊ per opener, distinct channels, the Thm 5 bound, and how often Thm 3's condition ‖E_S‖ < γ/d holds.

Usage: python amp_mode_bound_table.py
Output: downstream_amp/checks/amp_mode_bound_table.log
"""
import numpy as np
import torch

from amp_common import RUNS, log_to

log_to("amp_mode_bound_table")
MLP = RUNS / "mode_bound" / "mlp"
rows = []
for p in sorted((RUNS / "mode_bound").glob("*.pt")):
    if p.name.startswith(("owt_", "capacity_")):
        continue
    r = torch.load(p, weights_only=False)
    name = r["model"]
    ck = name.split("+", 1)[1] if "+" in name else None
    if ck:
        c = torch.load(MLP / f"{ck}.pt", weights_only=False)
        w, act, opt, seed = ck.split("_")
        rec = (c["none"] - c["held"]) / (c["none"] - c["pretrained"])
    else:
        act, opt, seed, rec = "gelu", "pretrained", "-", 1.0
    rows.append((r["H"], act, opt, seed, rec, r, p.stem))
rows.sort(key=lambda t: (t[2] == "pretrained", t[1], t[2], t[0], t[3]))
gammas = sorted(rows[0][5]["summary"])
print("per γ: neurons with a mode (k = 1) / max N₊ per opener / distinct channels / Thm 5 bound (spectral, B as measured) / "
      "Thm 3 holds for greedy sets")
print(f"{'run':44s} {'d_mlp':>5s} {'act':5s} {'opt':10s} {'seed':4s} {'recov':>5s} {'attr':>4s} {'κ':>6s} {'λ₁(K)':>6s} | " +
      " | ".join(f"γ = {g:g}".ljust(26) for g in gammas))
for H, act, opt, seed, rec, r, stem in rows:
    cells = []
    for g in gammas:
        s = r["summary"][g]
        cells.append(f"{s['openers1']:4d} {s['Npos1_max'] if s['openers1'] else 0:2d} {s.get('channels', 0):3d} "
                     f"{s.get("thm5", "-")!s:>4s} {s.get('greedy_thm3', 0):2d}/{s.get('greedy_n', 0):<2d}".ljust(26))
    print(f"{stem[:44]:44s} {H:5d} {act:5s} {opt:10s} {seed:4s} {rec:5.2f} {r['pipeline_attractors']:4d} {r['kappa']:6.3f} "
          f"{r['lamK'][0]:6.4f} | " + " | ".join(cells))

print("\nd_mlp scaling (gelu, adamw; mean over seeds): distinct channels and pipeline attractors")
for g in gammas:
    by = {}
    for H, act, opt, seed, rec, r, stem in rows:
        if act == "gelu" and opt == "adamw":
            by.setdefault(H, []).append((r["summary"][g].get("channels", 0), r["pipeline_attractors"], r["summary"][g]["openers1"]))
    print(f"   γ = {g:g}: " + ", ".join(f"d_mlp {H}: channels {np.mean([v[0] for v in vs]):.1f}, attractors "
                                         f"{np.mean([v[1] for v in vs]):.1f}, openers {np.mean([v[2] for v in vs]):.0f}"
                                         for H, vs in sorted(by.items())))
