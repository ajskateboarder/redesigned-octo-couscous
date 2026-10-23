import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT.parent), str(ROOT)]

import numpy as np
import torch
from scipy.sparse.csgraph import connected_components
from torch.nn import functional as F

from bigram_mechanism import device, forward, ids, length, model, recurring_factors, start, tokenizer

mlp = model.transformer.h[8].mlp
left, right, down = (w.weight.float() for w in (mlp.Left, mlp.Right, mlp.Down))
tail, starts, steps, temperature, same = 0.1, 64, 300, 0.1, 0.9

with torch.no_grad():
    z = torch.cat([forward(*start(chunk))[1][:, 1:] for chunk in ids.split(256)]).reshape(-1, 1152)
    gate, up = mlp.left(z), mlp.right(z)
    sigmoid = torch.sigmoid(gate)
    swish, slope_up = gate * sigmoid, sigmoid * (1 + gate * (1 - sigmoid)) * up
centered = z - z.mean(0)


def transmission(signal):
    # sᵀ J(z) s at every token for each row s of signal (J = Jacobian of the MLP-8 output w.r.t. its input)
    a, b, c = signal @ left.T, signal @ right.T, signal @ down
    return slope_up @ (a * c).T + swish @ (b * c).T


def amp(opener, signal, soft=True):
    # E[sᵀJs | opener coordinate in its top 10%] − E[sᵀJs | rest]; soft threshold for gradient ascent
    standard = centered @ opener.T
    standard = (standard - standard.mean(0)) / standard.std(0)
    threshold = standard.detach().quantile(1 - tail, dim=0)
    weight = torch.sigmoid((standard - threshold) / temperature) if soft else (standard >= threshold).float()
    passed = transmission(signal)
    return (weight * passed).sum(0) / weight.sum(0) - ((1 - weight) * passed).sum(0) / (1 - weight).sum(0)


def ascend(cross, generator):
    picks = torch.randint(len(z), (2, starts), device=device, generator=generator)
    opener = F.normalize(centered[picks[0]], dim=-1).clone().requires_grad_()
    signal = F.normalize(centered[picks[1]], dim=-1).clone().requires_grad_()
    optimizer = torch.optim.Adam([opener, signal] if cross else [opener], lr=0.01)

    def pair():
        o = F.normalize(opener, dim=-1)
        if not cross:
            return o, o
        s = signal - (signal * o).sum(-1, keepdim=True) * o
        return o, F.normalize(s, dim=-1)

    for _ in range(steps):
        loss = -amp(*pair()).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        o, s = pair()
        return o, s, amp(o, s, soft=False)


def attractors(o, s, score):
    similar = torch.minimum(o @ o.T, s @ s.T) > same
    _, label = connected_components(similar.cpu().numpy(), directed=False)
    groups = {}
    for i, c in enumerate(label):
        groups.setdefault(c, []).append(i)
    return sorted(groups.values(), key=lambda g: (-len(g), -score[g].mean().item()))


def context(n):
    p, t = divmod(n, length - 1)
    t += 1
    return (tokenizer.decode(ids[p, max(0, t - 3):t].tolist()) + "[" + tokenizer.decode([ids[p, t].item()]) + "]").replace("\n", "⏎")


@torch.no_grad()
def passes(opener, signal):
    # mean MLP-8 output change per unit of signal on opened tokens, read through the unembedding
    opened = (centered @ opener) >= (centered @ opener).quantile(1 - tail)
    change = down @ (slope_up[opened].mean(0) * (left @ signal) + swish[opened].mean(0) * (right @ signal))
    write = model.lm_head(F.rms_norm(change, (1152,))).float()[:len(tokenizer)]
    write = write - write.mean()
    top = lambda w: " ".join(repr(tokenizer.decode([t])) for t in w.topk(6).indices.tolist())
    return f"promotes {top(write)} | suppresses {top(-write)}"


asymmetric, quadratic = recurring_factors(""), recurring_factors("quadratic_")
dct = {f"asym #{k} {key.lower()}": F.normalize(f[key].to(device), dim=0) for k, f in enumerate(asymmetric) for key in "LR"}
dct.update({f"quad #{k} v": F.normalize(f["L"].to(device), dim=0) for k, f in enumerate(quadratic)})


def nearest(direction):
    name, cos = max(((n, (d @ direction).abs().item()) for n, d in dct.items()), key=lambda x: x[1])
    return f"{name} ({cos:.2f})"


generator = torch.Generator(device=device).manual_seed(0)
found = {}
for cross in (False, True):
    o, s, score = ascend(cross, generator)
    groups = attractors(o, s, score)
    kind = "cross (opener ⟂ signal)" if cross else "self (the post's Amp)"
    found[cross] = [(o[g[0]], s[g[0]]) for g in groups if len(g) >= 2]
    print(f"\n{kind}: {starts} runs from random token activations → {len(groups)} distinct end points "
          f"(cos > {same}); {sum(len(g) >= 2 for g in groups)} reached by ≥ 2 runs, covering {sum(len(g) for g in groups if len(g) >= 2)} runs")
    for k, g in enumerate(groups[:8]):
        i = g[0]
        print(f"  attractor {k}: {len(g)} runs, score {score[g].mean():.1f}")
        print(f"    opener fires on: " + " | ".join(context(n) for n in (centered @ o[i]).topk(5).indices.tolist())
              + f"   nearest DCT input: {nearest(o[i])}")
        if cross:
            print(f"    signal fires on: " + " | ".join(context(n) for n in (centered @ s[i]).topk(5).indices.tolist())
                  + f"   nearest DCT input: {nearest(s[i])}")
        print(f"    what passes MLP 8 when opened: {passes(o[i], s[i])}")

print("\nhow close each method's inputs are to the recurring attractors (max |cos|, median over that method's input directions)")
for method in ("asym", "quad"):
    names = [n for n in dct if n.startswith(method)]
    for cross, label in ((False, "self attractors"), (True, "cross attractors (opener or signal)")):
        targets_ = torch.stack([d for pair in found[cross] for d in (pair if cross else pair[:1])]) if found[cross] else None
        closeness = [(targets_ @ dct[n]).abs().max().item() for n in names] if targets_ is not None else [float("nan")]
        print(f"  {method}: {label} {np.median(closeness):.2f}")
