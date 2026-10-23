"""Is a model's MLP input written in a privileged (basis-aligned) way? Fit-free, so it runs on any model.

Rows: positions 16-31 of the same OpenWebText windows downstream_amp.py fits on (row group 60, draw 0), z = the input of
MLP --layer (after the block's norm). For the Elriggs tensor_model.py family z is RMSNorm without weights, i.e. the
residual direction itself; for GPT-2 it is ln_2's output (affine kept, and also with the affine undone); for gpt2_noln
the mean-centred residual.

    token-class Δ   mean z on positions whose current token is in a class (".", ",", newline, " the", byte fragment "�",
                    ...) minus the mean on all other positions: a fit-free stand-in for the openers downstream_amp.py
                    finds, which were position types of this kind. Reported: coordinate participation ratio of Δ
                    (1/Σu⁴ for u = Δ/|Δ|: 1 if Δ is a single coordinate, ≈ D/3 for a random direction) and the share of
                    |Δ|² in its largest coordinate
    outlier coords  per-coordinate excess kurtosis (max, and how many exceed 10) and the variance share of the top 1% of
                    coordinates by variance
    top PCs         excess kurtosis of the top 16 PCs (90th pct, max) and the coordinate participation ratio of their
                    eigenvectors (median)
Training: GPT-2 small with Adam; gpt2_noln fine-tuned from it with AdamW (nanoGPT); every tensor_model.py model with Muon;
SmolLM-135M (Llama-style, SwiGLU, 30 layers; layers 13 and 20 match the others' relative depths) released July 2024,
before Muon existed. Its z is the post-attention RMSNorm output, also reported with the RMSNorm weight divided out.

Plot: excess kurtosis of the top 64 PCs by rank, one layer per model at matched relative depth (2/3: GPT-2 and 12L layer 8,
18L layer 12, SmolLM layer 20; raw MLP input), and the distribution of the coordinate along each model's most heavy-tailed
top-16 PC for GPT-2 (Adam, GELU) and the 12L ReLU² model (Muon), the closest activation-matched pair across optimizers.

Usage: python amp_basis_alignment.py [--windows 480]
Output: downstream_amp/checks/amp_basis_alignment.log, amp_basis_alignment_pc_kurtosis.{png,npz}
"""
import argparse
import gc
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT_DIR.parent), str(ROOT_DIR), str(ROOT_DIR / "cross_token")]

import numpy as np
import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download

import amp_models
from amp_common import CHECKS, log_to
from tensor_model import load_tensor_gpt

parser = argparse.ArgumentParser()
parser.add_argument("--windows", type=int, default=480)
cli = parser.parse_args()
log_to("amp_basis_alignment")
dev = torch.device("cuda")
WIN, FIRST = 32, 16
torch.set_grad_enabled(False)

OWT = hf_hub_download("apollo-research/Skylion007-openwebtext-tokenizer-gpt2", "data/train-00000-of-00073.parquet",
                      repo_type="dataset")
table, rng, out, spans = pq.ParquetFile(OWT).read_row_group(60), np.random.default_rng(0), [], []
for i in range(table.num_rows):
    seq = np.asarray(table.column(0)[i].as_py())
    if len(seq) > WIN + 8:
        o_ = int(rng.integers(0, len(seq) - WIN))
        if not (seq[o_:o_ + WIN] == 50256).any():
            out.append(seq[o_:o_ + WIN])
            spans.append(seq[o_:o_ + 2 * WIN])                  # longer span of the same text, for other tokenizers
    if len(out) == cli.windows:
        break
ids = torch.tensor(np.stack(out), device=dev)

from transformers import AutoTokenizer, GPT2Tokenizer
gpt2_tok = GPT2Tokenizer.from_pretrained("gpt2")


def token_classes(tok, cur):
    # positions whose current token is in each class, from the tokenizer's own decoding of every vocabulary entry
    dec = [tok.decode([i]) for i in range(len(tok))]
    classes = {
        '"."': [i for i, d in enumerate(dec) if d == "."],
        '","': [i for i, d in enumerate(dec) if d == ","],
        "newline": [i for i, d in enumerate(dec) if d and set(d) == {"\n"}],
        '" the"': [i for i, d in enumerate(dec) if d == " the"],
        '" of"/" to"/" and"': [i for i, d in enumerate(dec) if d in (" of", " to", " and")],
        "byte fragment": [i for i, d in enumerate(dec) if "\ufffd" in d],
        "capitalised word": [i for i, d in enumerate(dec) if len(d) > 2 and d[0] == " " and d[1].isupper() and d[1:].isalpha()],
        "digit": [i for i, d in enumerate(dec) if d.strip().isdigit()],
    }
    return {k: torch.isin(cur, torch.tensor(v, device=dev)) for k, v in classes.items() if v}


member = token_classes(gpt2_tok, ids[:, FIRST:].flatten())


def tensor_inputs(repo, layers):
    model, _, _ = load_tensor_gpt(repo, dev)
    model.requires_grad_(False)
    got = {}
    hooks = [model.transformer.h[L].mlp.register_forward_pre_hook(
        lambda m, a, L=L: got.setdefault(L, []).append(a[0][:, FIRST:].float())) for L in layers]
    for s in range(0, len(ids), 64):
        model.logits(ids[s:s + 64])
    for h in hooks:
        h.remove()
    del model
    return {L: torch.cat(v).reshape(-1, v[0].shape[-1]) for L, v in got.items()}


def adapter_inputs(name, layers, undo_affine=False):
    M = amp_models.load(name, WIN)
    res = {}
    for L in layers:
        z = torch.cat([M.capture(ids[s:s + 64], L)[0][:, FIRST:] for s in range(0, len(ids), 64)]).reshape(-1, M.D)
        if undo_affine:
            ln2 = M.hf.transformer.h[L].ln_2
            z = (z - ln2.bias) / ln2.weight
        res[L] = z.float()
    del M
    return res


def hf_inputs(repo, layers, undo_weight=False):
    # Llama-style Hugging Face model: the same documents re-tokenised with the model's own tokenizer (first WIN tokens of a
    # 2·WIN-token GPT-2 span); z = the MLP's input (post-attention RMSNorm output, weight kept or divided out)
    from transformers import AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.float32).to(dev).eval()
    rows = [r for r in (tok(gpt2_tok.decode(s), add_special_tokens=False).input_ids for s in spans) if len(r) >= WIN]
    own = torch.tensor([r[:WIN] for r in rows], device=dev)
    layer_mods = model.model.layers
    got = {}
    hooks = [layer_mods[L].mlp.register_forward_pre_hook(
        lambda m, a, L=L: got.setdefault(L, []).append(a[0][:, FIRST:].float())) for L in layers]
    for s in range(0, len(own), 64):
        model(own[s:s + 64])
    for h in hooks:
        h.remove()
    res = {}
    for L in layers:
        z = torch.cat(got[L]).reshape(-1, model.config.hidden_size)
        res[L] = z / layer_mods[L].post_attention_layernorm.weight.float() if undo_weight else z
    del model
    return res, token_classes(tok, own[:, FIRST:].flatten())


def report(label, z, member=member):
    D = z.shape[1]
    zc = z - z.mean(0)
    var = zc.var(0)
    kc = (zc ** 4).mean(0) / var.square() - 3
    top1 = var.sort(descending=True).values[:max(1, D // 100)].sum() / var.sum()
    C = torch.cov(zc.T.double())
    ev, evec = torch.linalg.eigh(C)
    E64 = evec.flip(1)[:, :64].float()
    pc64 = zc @ E64
    kp64 = (pc64 ** 4).mean(0) / pc64.var(0).square() - 3
    E, pc, kp = E64[:, :16], pc64[:, :16], kp64[:16]
    pr = lambda u: (1 / (u / u.norm(dim=0)).pow(4).sum(0))
    pr_pc = pr(E).median().item()
    prs, tops = [], []
    for k, m in member.items():
        if m.sum() < 20:
            continue
        delta = z[m].mean(0) - z[~m].mean(0)
        prs.append(pr(delta[:, None]).item())
        tops.append((delta.square().max() / delta.square().sum()).item())
    prs, tops = np.array(prs), np.array(tops)
    print(f"  {label:<34} D {D:>4} | token-class Δ: coord PR median {np.median(prs):6.0f} (min {prs.min():5.0f}; random ≈ {D / 3:.0f}),"
          f" top-coord share median {np.median(tops):.2f} | coord kurtosis max {kc.max():7.1f}, >10: {int((kc > 10).sum()):>3},"
          f" top-1% var share {top1:.2f} | top-16 PC kurtosis 90th pct {kp.quantile(.9):5.2f}, max {kp.max():5.2f},"
          f" eigvec coord PR median {pr_pc:4.0f}", flush=True)
    top = int(kp.argmax())
    c = pc[:, top]
    c = (c - c.mean()) / c.std()
    c = c * torch.sign((c ** 3).mean())                         # long tail to the right
    return dict(label=label, pr=float(np.median(prs)), D=D, kurt=kp64.cpu().numpy(), coord=c.cpu().numpy(), top_pc=top)


print(f"{len(ids)} windows × {WIN - FIRST} positions; token classes with ≥ 20 positions: "
      + ", ".join(f"{k} ({int(m.sum())})" for k, m in member.items() if m.sum() >= 20))
runs = [
    ("Adam  GPT-2 small (GELU)", lambda: adapter_inputs("gpt2", [5, 8]), [5, 8]),
    ("Adam  GPT-2 small, ln_2 affine undone", lambda: adapter_inputs("gpt2", [5, 8], undo_affine=True), [5, 8]),
    ("AdamW GPT-2 noLN (GELU, fine-tuned)", lambda: adapter_inputs("gpt2_noln", [5, 8]), [5, 8]),
    ("Muon  12L ReLU²", lambda: tensor_inputs("Elriggs/gpt2-12l-6h-768embd", [5, 8]), [5, 8]),
    ("Muon  12L x²", lambda: tensor_inputs("Elriggs/gpt2-mlp2-12l-6h-768embd", [5, 8]), [5, 8]),
    ("Muon  12L bilinear", lambda: tensor_inputs("Elriggs/gpt2-bilinear-12l-6h-768embd", [5, 8]), [5, 8]),
    ("Muon  12L SwiGLU", lambda: tensor_inputs("Elriggs/gpt2-bilinear-swiglu-12l-6h-768embd", [5, 8]), [5, 8]),
    ("Muon  18L bilinear", lambda: tensor_inputs("Elriggs/gpt2-bilinear-18l-9h-1152embd", [8, 12]), [8, 12]),
    ("Muon  18L SwiGLU (the one fitted)", lambda: tensor_inputs("Elriggs/gpt2-swiglu-18l-9h-1152embd-v2", [8, 12]), [8, 12]),
]
runs += [
    ("AdamW SmolLM-135M (Llama, SwiGLU)", lambda: hf_inputs("HuggingFaceTB/SmolLM-135M", [13, 20]), [13, 20]),
    ("AdamW SmolLM-135M, RMSNorm weight undone", lambda: hf_inputs("HuggingFaceTB/SmolLM-135M", [13, 20], undo_weight=True),
     [13, 20]),
]
PLOT = {"Adam  GPT-2 small (GELU)": ("GPT-2 (Adam, GELU)", 8), "AdamW GPT-2 noLN (GELU, fine-tuned)": ("GPT-2 noLN (AdamW, GELU)", 8),
        "AdamW SmolLM-135M (Llama, SwiGLU)": ("SmolLM-135M (AdamW, SwiGLU)", 20), "Muon  12L ReLU²": ("12L ReLU² (Muon)", 8),
        "Muon  12L x²": ("12L x² (Muon)", 8), "Muon  12L bilinear": ("12L bilinear (Muon)", 8),
        "Muon  12L SwiGLU": ("12L SwiGLU (Muon)", 8), "Muon  18L bilinear": ("18L bilinear (Muon)", 12),
        "Muon  18L SwiGLU (the one fitted)": ("18L SwiGLU (Muon)", 12)}
plotted = {}
for name, get, layers in runs:
    print(name, flush=True)
    zs = get()
    zs, mem = zs if isinstance(zs, tuple) else (zs, member)
    for L in layers:
        out = report(f"layer {L}", zs[L], mem)
        if name in PLOT and PLOT[name][1] == L:
            plotted[PLOT[name][0]] = out
    del zs
    gc.collect()
    torch.cuda.empty_cache()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.savez(CHECKS / "amp_basis_alignment_pc_kurtosis.npz",
         **{f"kurt:{k}": v["kurt"] for k, v in plotted.items()}, **{f"coord:{k}": v["coord"] for k, v in plotted.items()})
fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 5.4), gridspec_kw={"width_ratios": [1.6, 1]})
warm, cool = plt.cm.autumn(np.linspace(0, 0.7, 3)), plt.cm.winter(np.linspace(0, 1, 6))
colors, iw, ic = {}, 0, 0
for k in plotted:
    if "Muon" in k:
        colors[k], ic = cool[ic], ic + 1
    else:
        colors[k], iw = warm[iw], iw + 1
ranks = np.arange(1, 65)
for k, v in plotted.items():
    ax.plot(ranks, v["kurt"], color=colors[k], lw=2.2 if "Muon" not in k else 1.4, ls="-" if "Muon" not in k else "--",
            marker="o", ms=3, label=k)
ax.axhline(0, color="k", lw=0.8)
ax.axvspan(0.5, 16.5, color="0.92", zorder=0)
ax.set_xscale("log")
ax.set_xticks([1, 2, 4, 8, 16, 32, 64])
ax.set_xticklabels(["1", "2", "4", "8", "16", "32", "64"])
ax.set_xlabel("principal component rank (MLP input, 2/3 depth)")
ax.set_ylabel("excess kurtosis (Gaussian = 0)")
ax.set_title("How heavy-tailed each principal component is")
ax.legend(fontsize=7.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.17), frameon=False)
bins = np.linspace(-4, 6, 61)
xs = 0.5 * (bins[1:] + bins[:-1])
for k in ("GPT-2 (Adam, GELU)", "12L ReLU² (Muon)"):
    if k in plotted:
        h, _ = np.histogram(plotted[k]["coord"], bins=bins, density=True)
        bx.semilogy(xs, np.where(h > 0, h, np.nan), color=colors[k], lw=2,
                    label=f"{k}, PC {plotted[k]['top_pc'] + 1} (kurtosis {plotted[k]['kurt'][plotted[k]['top_pc']]:.1f})")
bx.semilogy(xs, np.exp(-xs ** 2 / 2) / np.sqrt(2 * np.pi), color="k", lw=0.8, ls=":", label="Gaussian")
bx.set_ylim(1e-4, 1)
bx.set_xlabel("coordinate (standardised)")
bx.set_ylabel("density")
bx.set_title("Most heavy-tailed of the top 16 PCs")
bx.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.17), frameon=False)
fig.tight_layout()
fig.savefig(CHECKS / "amp_basis_alignment_pc_kurtosis.png", dpi=160)
print(f"plot: {CHECKS / 'amp_basis_alignment_pc_kurtosis.png'}")
