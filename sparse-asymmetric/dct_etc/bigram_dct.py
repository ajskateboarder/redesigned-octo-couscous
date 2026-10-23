import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
for path in (str(ROOT.parent), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import GPT2Tokenizer

from bilinear_attention_pr_experiment import AttentionRegularizedBilinearDCT, SlicedModel, cross_seed_alignment
from intermediate_mlp_pr_experiment import read_texts
from tensor_model import load_tensor_gpt

length, targets, contexts, top_k, seeds = 12, [4, 8, 11], 32, 64, range(5)
out = ROOT / "natural_dct" / "bigram"
out.mkdir(parents=True, exist_ok=True)
device = torch.device("cuda")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model, _, _ = load_tensor_gpt("Elriggs/gpt2-swiglu-18l-9h-1152embd-v2", device)
model.requires_grad_(False)
vocab = len(tokenizer)

chain_path = out / "chain.pt"
if chain_path.exists():
    next_ids, next_probs = torch.load(chain_path)
else:
    # the model's own bigram: next-token distribution from each single token, truncated to its top-k continuations
    next_ids, next_probs = [], []
    with torch.no_grad():
        for batch in torch.arange(vocab, device=device).split(2048):
            probs = model.logits(batch[:, None])[:, 0, :vocab].softmax(-1)
            top = probs.topk(top_k, dim=-1)
            next_ids.append(top.indices)
            next_probs.append(top.values / top.values.sum(-1, keepdim=True))
    next_ids, next_probs = torch.cat(next_ids), torch.cat(next_probs)
    torch.save((next_ids, next_probs), chain_path)


def sample_chain(count, seed):
    generator = torch.Generator(device=device).manual_seed(seed)
    token = torch.full((count,), tokenizer.eos_token_id, device=device)
    sequence = []
    for _ in range(length):
        choice = torch.multinomial(next_probs[token], 1, generator=generator)[:, 0]
        token = next_ids[token, choice]
        sequence.append(token)
    return torch.stack(sequence, 1)


def state_of(ids):
    with torch.no_grad():
        start = model.state_before_block(ids, 8)
    return tuple(start[key].float() for key in ("values", "initial_values", "first_values"))


natural = [t for t in read_texts(0, 520)[1] if len(tokenizer(t).input_ids) >= length][:contexts]
datasets = {
    "bigram": sample_chain(contexts, 0),
    "natural": torch.tensor([tokenizer(t).input_ids[:length] for t in natural], device=device),
}
print("bigram samples:", *[repr(tokenizer.decode(s)) for s in datasets["bigram"][:4].tolist()], sep="\n  ")

fits = {}
for name, ids in datasets.items():
    state = state_of(ids)
    delta = SlicedModel(model, state, 8, 12, targets)
    zero = torch.zeros(1152, device=device)
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
        clean = delta(zero, zero, state[0], torch.zeros_like(state[0]))[0]
    for seed in seeds:
        path = out / f"{name}_seed{seed}.pt"
        if not path.exists():
            with sdpa_kernel(SDPBackend.MATH):
                dct = AttentionRegularizedBilinearDCT(16, 0.0, 1.0)
                dct.fit(delta, state[0], clean, 8, 16, 100, seed=seed)
            torch.save({k: getattr(dct, k).detach().float().cpu() for k in "ULR"}, path)
        saved = torch.load(path)
        fits[name, seed] = SimpleNamespace(**{k: F.normalize(v, dim=0) for k, v in saved.items()})
        print(f"fitted {name} seed {seed}", flush=True)

print("\nreproducibility: factors matched one-to-one between fits, similarity = |cos u|·|cos l|·|cos r|; reproduced = similarity > 0.8")
for left, right in (("bigram", "bigram"), ("natural", "natural"), ("bigram", "natural")):
    pairs = [(a, b) for a in seeds for b in seeds if (a < b if left == right else a == b)]
    matched = [cross_seed_alignment(fits[left, a], fits[right, b]) for a, b in pairs]
    counts = [m["matches_above_0_8"] for m in matched]
    print(f"  {left:7s} vs {right:7s}: reproduced factors per pair of fits {np.mean(counts):4.1f}/16 "
          f"(range {min(counts)}-{max(counts)}), median matched similarity {np.median([m['median_similarity'] for m in matched]):.2f}")

reference = fits["bigram", 0]
support = np.zeros(16, dtype=int)
natural_match = np.zeros(16)
for seed in seeds[1:]:
    similarity = ((reference.U.T @ fits["bigram", seed].U).abs() * (reference.L.T @ fits["bigram", seed].L).abs()
                  * (reference.R.T @ fits["bigram", seed].R).abs())
    support += (similarity.max(1).values > 0.8).numpy()
for seed in seeds:
    similarity = ((reference.U.T @ fits["natural", seed].U).abs() * (reference.L.T @ fits["natural", seed].L).abs()
                  * (reference.R.T @ fits["natural", seed].R).abs())
    natural_match = np.maximum(natural_match, similarity.max(1).values.numpy())
stable = [i for i in range(16) if support[i] >= 3]
print(f"\nbigram seed-0 factors found again (>0.8) in at least 3 of the 4 other bigram seeds: {stable}")

held = sample_chain(512, 1)
with torch.no_grad():
    start = model.state_before_block(held, 8)
    block = model.transformer.h[8]
    values = block.lambdas[0] * start["values"] + block.lambdas[1] * start["initial_values"]
    attention, _ = block.attn(F.rms_norm(values, (1152,)), start["first_values"])
    inputs = F.rms_norm(values + attention, (1152,))[:, 1:].reshape(-1, 1152)
centered = inputs - inputs.mean(0)
sites = [(p, t) for p in range(len(held)) for t in range(1, length)]
context = lambda n: (lambda p, t: f"{tokenizer.decode(held[p, max(0, t - 3):t].tolist())}[{tokenizer.decode([held[p, t].item()])}]".replace("\n", "⏎"))(*sites[n])
U, Lm, Rm = (getattr(reference, k).to(device) for k in "ULR")
for i in stable:
    # logit lens of the layer-12 output direction
    write = model.lm_head(F.rms_norm(U[:, i], (1152,))).float()[:vocab]
    print(f"\nfactor {i}: stable in {support[i]}/4 other bigram seeds; best match in natural fits {natural_match[i]:.2f}")
    print("  l reads:", " | ".join(context(n) for n in (centered @ Lm[:, i]).abs().topk(6).indices.tolist()))
    print("  r reads:", " | ".join(context(n) for n in (centered @ Rm[:, i]).abs().topk(6).indices.tolist()))
    print("  u writes ±:", " ".join(repr(tokenizer.decode([t])) for t in write.topk(6).indices.tolist()), "|",
          " ".join(repr(tokenizer.decode([t])) for t in (-write).topk(6).indices.tolist()))
