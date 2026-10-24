"""Retrain gelu_1l's MLP at other widths, activations, optimizers and seeds: the d_mlp scaling test at fixed d_model = 512
(gate_bound.md step 7) and the robustness runs (step 8). Embeddings, attention, LayerNorms and unembedding are the
pretrained NeelNanda/GELU_1L512W_C4_Code and stay frozen; only the MLP is trained, from scratch, on next-token loss on
OpenWebText row groups 0-59 (downstream_amp.py's rows come from 60-61), retokenised into 64-token windows that start with
<|BOS|>. Every run sees the same windows in the same order (one epoch), so width, activation and optimizer are the only
differences. Held-out loss is on row group 63, next to the pretrained MLP's and to no MLP at all.

    adamw  lr 2e-3, betas (0.9, 0.95), weight decay 0.1 on the matrices, 2% warmup, cosine decay
    muon   matrices: Nesterov momentum 0.95, update orthogonalised by 5 Newton-Schulz steps, lr 0.02·√max(1, rows/cols);
           biases: the adamw settings

Usage: python amp_mlp_width.py --width 2048 [--act gelu|relu|relu2] [--opt adamw|muon] [--seed 0]
Output: downstream_amp/gate_bound/mlp/w{width}_{act}_{opt}_s{seed}.pt, used as --model gelu_1l+w{width}_{act}_{opt}_s{seed}
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download
from torch.nn import functional as F

ROOT_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT_DIR)]
import amp_models

parser = argparse.ArgumentParser()
parser.add_argument("--width", type=int, default=2048)
parser.add_argument("--act", choices=list(amp_models.ACTS), default="gelu")
parser.add_argument("--opt", choices=["adamw", "muon"], default="adamw")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--batch", type=int, default=256, help="windows per step")
args = parser.parse_args()

WIN = 64
OUT = amp_models.RETRAINED
CACHE = OUT.parent / "owt_gelu_1l_win64.pt"
torch.set_float32_matmul_precision("high")
M = amp_models.load("gelu_1l", WIN)
dev, D, sd = M.dev, M.D, M.sd


def windows(groups):
    # each parquet row is 1024 GPT-2 tokens of concatenated documents; split at <|endoftext|>, retokenise each document
    from transformers import GPT2Tokenizer
    gpt2 = GPT2Tokenizer.from_pretrained("gpt2")
    path = hf_hub_download("apollo-research/Skylion007-openwebtext-tokenizer-gpt2", "data/train-00000-of-00073.parquet",
                           repo_type="dataset")
    f, out = pq.ParquetFile(path), []
    for g in groups:
        docs = []
        for seq in f.read_row_group(g).column(0).to_pylist():
            seq = np.asarray(seq)
            docs += [d for d in np.split(seq, np.nonzero(seq == 50256)[0]) if len(d) > 8]
        texts = gpt2.batch_decode([d[d != 50256] for d in docs])
        for ids in M.tokenizer(texts, add_special_tokens=False).input_ids:
            for s in range(0, len(ids) - (WIN - 1) + 1, WIN - 1):
                out.append([M.bos] + ids[s:s + WIN - 1])
    return torch.tensor(out, dtype=torch.int32)


if not CACHE.exists():
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    torch.save({"train": windows(range(60)), "eval": windows([63])}, CACHE)
    print(f"tokenised windows cached in {CACHE} ({time.time() - t0:.0f} s)")
data = torch.load(CACHE)
train, evalw = data["train"], data["eval"][:2048]

torch.manual_seed(args.seed)
H = args.width
lin_in, lin_out = torch.nn.Linear(D, H, device=dev), torch.nn.Linear(H, D, device=dev)
params = dict(W_in=lin_in.weight.T.detach().clone(), b_in=lin_in.bias.detach().clone(),
              W_out=lin_out.weight.T.detach().clone(), b_out=lin_out.bias.detach().clone())
for p in params.values():
    p.requires_grad_()
act = amp_models.ACTS[args.act]


def loss_of(ids, mlp=True):
    x, r, _ = M.capture(ids.long().to(dev), 0)
    if mlp is True:
        h = r + act(x @ params["W_in"] + params["b_in"]) @ params["W_out"] + params["b_out"]
    else:
        h = r + (0 if mlp is None else M.downstream(x, r, (), 0.0, 0) - r)
    with torch.autocast("cuda", torch.bfloat16):
        logits = M.logits(h)
    return F.cross_entropy(logits[:, :-1].float().flatten(0, 1), ids[:, 1:].long().to(dev).flatten())


def newton_schulz(G, steps=5):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    X = X.T if G.shape[0] > G.shape[1] else X
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    return (X.T if G.shape[0] > G.shape[1] else X).float()


mats, vecs = [params["W_in"], params["W_out"]], [params["b_in"], params["b_out"]]
steps = len(train) // args.batch
sched = lambda i: min(1.0, i / (0.02 * steps)) * 0.5 * (1 + math.cos(math.pi * i / steps))
if args.opt == "adamw":
    opt = torch.optim.AdamW([{"params": mats, "weight_decay": 0.1}, {"params": vecs, "weight_decay": 0.0}], lr=2e-3,
                            betas=(0.9, 0.95))
else:
    opt = torch.optim.AdamW(vecs, lr=2e-3, betas=(0.9, 0.95), weight_decay=0.0)
    bufs = [torch.zeros_like(p) for p in mats]
order = torch.randperm(len(train), generator=torch.Generator().manual_seed(1234))       # the same for every run
t0, log = time.time(), []
for i in range(steps):
    ids = train[order[i * args.batch:(i + 1) * args.batch]]
    loss = loss_of(ids)
    opt.zero_grad()
    for p in mats:
        p.grad = None
    loss.backward()
    for g in opt.param_groups:
        g["lr"] = 2e-3 * sched(i)
    opt.step()
    if args.opt == "muon":
        with torch.no_grad():
            for p, buf in zip(mats, bufs):
                buf.mul_(0.95).add_(p.grad)
                p -= 0.02 * sched(i) * max(1, p.shape[0] / p.shape[1]) ** 0.5 * newton_schulz(p.grad + 0.95 * buf)
    log.append(loss.item())
    if (i + 1) % 500 == 0 or i == steps - 1:
        print(f"  step {i + 1}/{steps}: train loss {np.mean(log[-100:]):.4f} ({time.time() - t0:.0f} s)", flush=True)

with torch.no_grad():
    ev = lambda mlp: np.mean([loss_of(evalw[s:s + 128], mlp).item() for s in range(0, len(evalw), 128)])
    held, pre, none = ev(True), ev(False), ev(None)
name = f"w{H}_{args.act}_{args.opt}_s{args.seed}"
print(f"{name}: held-out loss {held:.4f} | pretrained MLP {pre:.4f} | no MLP {none:.4f} | share of the MLP's loss reduction "
      f"recovered {(none - held) / (none - pre):.3f} ({len(train) * (WIN - 1) / 1e6:.0f}M tokens, {time.time() - t0:.0f} s)")
OUT.mkdir(parents=True, exist_ok=True)
torch.save({**{k: v.detach() for k, v in params.items()}, "act": args.act, "args": vars(args), "held": held,
            "pretrained": pre, "none": none, "train_loss": log}, OUT / f"{name}.pt")
print(f"saved {OUT / name}.pt")
