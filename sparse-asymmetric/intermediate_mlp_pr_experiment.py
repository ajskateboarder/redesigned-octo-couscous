import sys
import os
from pathlib import Path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(current_dir))

import argparse
import csv
import itertools
import json
import random

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import GPT2Tokenizer

from dct import AsymmetricQuadraticDCT, directional_hessian_outputs
from tensor_model import TensorMiddleSpan, load_tensor_gpt


ROOT = Path(__file__).resolve().parent
REPOSITORIES = {
    "bilinear": "Elriggs/gpt2-bilinear-18l-9h-1152embd",
    "swiglu": "Elriggs/gpt2-swiglu-18l-9h-1152embd-v2",
    "bilinear-attn": "Elriggs/gpt2-bilinear-sqrd-attn-18l-9h-1152embd",
    "swiglu-attn": "Elriggs/gpt2-swiglu-sqrd-attn-18l-9h-1152embd",
}


def read_texts(train_count, heldout_count):
    with (ROOT / ".." / "harmful_behaviors.csv").open() as handle:
        texts = list(dict.fromkeys(row["target"].strip() for row in csv.DictReader(handle)))
    random.Random(1729).shuffle(texts)
    if train_count + heldout_count > len(texts):
        raise ValueError("requested more unique texts than are available")
    return texts[:train_count], texts[train_count:train_count + heldout_count]


def collect_middle_inputs(model, tokenizer, texts, source_layer, sequence_length):
    values, initial_values, first_values = [], [], []
    for text in texts:
        token_ids = tokenizer(
            text, return_tensors="pt", truncation=True, padding="max_length",
            max_length=sequence_length,
        ).input_ids.to(next(model.parameters()).device)
        with torch.no_grad():
            state = model.state_before_block(token_ids, source_layer)
        values.append(state["values"].float())
        initial_values.append(state["initial_values"].float())
        first_values.append(state["first_values"].float())
    return torch.cat(values), torch.cat(initial_values), torch.cat(first_values)


class MiddleSpanOperator(nn.Module):
    def __init__(self, model, state, source_layer, target_layer):
        super().__init__()
        self.span = TensorMiddleSpan(model, source_layer, target_layer)
        self.register_buffer("reference_values", state[0])
        self.register_buffer("initial_values", state[1])
        self.register_buffer("first_values", state[2])

    def context_indices(self, values):
        distances = (self.reference_values[:, None] - values[None, :]).square().flatten(2).sum(2)
        return distances.argmin(dim=0)

    def forward(self, values):
        indices = self.context_indices(values)
        return self.span(
            values, self.initial_values[indices], self.first_values[indices],
        )


class SpanDelta(nn.Module):
    def __init__(self, operator, target_positions=slice(-3, None)):
        super().__init__()
        self.operator = operator
        self.device = operator.reference_values.device
        self.dtype = operator.reference_values.dtype
        self.target_positions = target_positions

    def forward(self, theta, values, clean):
        output = self.operator(values + theta)
        return (output - clean)[:, self.target_positions].mean(dim=1)


class IntermediateMLPFeatures(nn.Module):
    def __init__(self, model, state, source_layer, target_layer, target_positions):
        super().__init__()
        self.blocks = model.transformer.h[source_layer:target_layer]
        self.source_layer = source_layer
        self.target_layer = target_layer
        self.target_positions = target_positions
        self.register_buffer("reference_values", state[0])
        self.register_buffer("initial_values", state[1])
        self.register_buffer("first_values", state[2])
        self.device = state[0].device
        self.dtype = state[0].dtype

    @property
    def intermediate_layers(self):
        return list(range(self.source_layer + 1, self.target_layer))

    def context_indices(self, values):
        distances = (self.reference_values[:, None] - values[None, :]).square().flatten(2).sum(2)
        return distances.argmin(dim=0)

    @staticmethod
    def hidden_features(mlp, values):
        if hasattr(mlp, "Left"):
            left = mlp.Left(values)
            if mlp.config.gated:
                left = F.silu(left)
            return left * mlp.Right(values)
        hidden = mlp.c_fc(values)
        return hidden.square() if mlp.config.squared_mlp else F.relu(hidden).square()

    def forward(self, theta, values, clean):
        indices = self.context_indices(values)
        values = values + theta
        initial_values = self.initial_values[indices]
        first_values = self.first_values[indices]
        features = []
        for relative_layer, block in enumerate(self.blocks):
            absolute_layer = self.source_layer + relative_layer
            values = block.lambdas[0] * values + block.lambdas[1] * initial_values
            attention, first_values = block.attn(
                F.rms_norm(values, (values.shape[-1],)), first_values,
            )
            values = values + attention
            mlp_input = F.rms_norm(values, (values.shape[-1],))
            hidden = self.hidden_features(block.mlp, mlp_input)
            if self.source_layer < absolute_layer < self.target_layer:
                features.append(hidden[:, self.target_positions].mean(dim=1))
            values = values + block.mlp(mlp_input)
        if not features:
            raise ValueError("the exclusive layer range contains no MLP layers")
        return torch.cat(features, dim=1)


def derangements(size, limit, generator):
    candidates = [
        permutation
        for permutation in itertools.permutations(range(size))
        if all(index != value for index, value in enumerate(permutation))
    ]
    if len(candidates) > limit:
        selected = torch.randperm(len(candidates), generator=generator)[:limit].tolist()
        candidates = [candidates[index] for index in selected]
    return [torch.tensor(permutation) for permutation in candidates]


def participation_ratio(response: torch.Tensor, epsilon=1e-30):
    energy = response.float().square()
    return energy.sum(dim=-1).square() / energy.square().sum(dim=-1).clamp_min(epsilon)


def pair_response_grid(feature_operator, values, factors_left, factors_right, pair_batch):
    factors = factors_left.shape[1]
    pair_left = factors_left.repeat_interleave(factors, dim=1)
    pair_right = factors_right.repeat(1, factors)
    with sdpa_kernel(SDPBackend.MATH):
        pair_responses = torch.vmap(
            lambda left, right: directional_hessian_outputs(
                feature_operator, left, right, values,
                torch.empty(0, device=values.device),
            ),
            in_dims=(1, 1), out_dims=1, chunk_size=pair_batch,
        )(pair_left, pair_right)
    return pair_responses.unflatten(1, (factors, factors))


def summarize_pr(response_grid, permutations, seed):
    contexts, factors = response_grid.shape[:2]
    if response_grid.shape[2] != factors:
        raise ValueError("pair response grid must have equally sized left and right axes")
    pair_pr = participation_ratio(response_grid)
    factor_indices = torch.arange(factors, device=response_grid.device)
    matched = pair_pr[:, factor_indices, factor_indices]
    observed = float(matched.median())

    generator = torch.Generator().manual_seed(seed)
    null_medians = []
    shuffled_values = []
    controls = derangements(factors, permutations, generator)
    for permutation in controls:
        permutation = permutation.to(response_grid.device)
        shuffled = pair_pr[:, factor_indices, permutation]
        null_medians.append(float(shuffled.median()))
        shuffled_values.append(shuffled)
    null_medians_tensor = torch.tensor(null_medians)
    shuffled_values_tensor = torch.stack(shuffled_values)
    p_value = float((1 + (null_medians_tensor <= observed).sum()) / (len(controls) + 1))
    null_median = float(null_medians_tensor.median())
    return {
        "matched_participation_ratio": matched.detach().cpu().tolist(),
        "matched_median_participation_ratio": observed,
        "shuffled_median_participation_ratio": null_median,
        "matched_to_shuffled_median_ratio": observed / max(null_median, 1e-30),
        "lower_tail_permutation_p_value": p_value,
        "unique_derangement_controls": len(controls),
        "shuffled_factor_context_pr_median": float(shuffled_values_tensor.median()),
        "contexts": contexts,
    }


def fit_dictionary(model, state, source_layer, target_layer, seed, factors, iterations, factor_batch):
    operator = MiddleSpanOperator(model, state, source_layer, target_layer).to(state[0].device)
    clean = operator(state[0]).detach()
    delta = SpanDelta(operator)
    torch.manual_seed(seed)
    dictionary = AsymmetricQuadraticDCT(num_factors=factors)
    with sdpa_kernel(SDPBackend.MATH):
        dictionary.fit(
            delta, state[0], clean, batch_size=16, factor_batch_size=factor_batch,
            max_iters=iterations,
        )
    return dictionary


def run_architecture(args, architecture, tokenizer, texts):
    model, config, metadata = load_tensor_gpt(REPOSITORIES[architecture], args.device)
    model.requires_grad_(False)
    states = [
        collect_middle_inputs(model, tokenizer, split, args.source_layer, args.sequence_length)
        for split in texts
    ]
    fits = []
    for fit_index, seed in enumerate(args.fit_seeds):
        dictionary = fit_dictionary(
            model, states[0], args.source_layer, args.target_layer, seed,
            args.factors, args.iterations, args.factor_batch,
        )
        split_results = {}
        for split_name, state in zip(("train", "heldout"), states):
            feature_operator = IntermediateMLPFeatures(
                model, state, args.source_layer, args.target_layer,
                slice(-args.target_positions, None),
            )
            response_grid = pair_response_grid(
                feature_operator, state[0], dictionary.L, dictionary.R, args.pair_batch,
            )
            summary = summarize_pr(
                response_grid, args.permutations, args.null_seed + fit_index,
            )
            summary["intermediate_layers"] = feature_operator.intermediate_layers
            summary["feature_width_per_layer"] = config.expansion_factor * config.n_embd
            summary["total_feature_coordinates"] = response_grid.shape[-1]
            split_results[split_name] = summary
        fits.append({"fit_seed": seed, "splits": split_results})
    return {
        "repository": REPOSITORIES[architecture],
        "model_metadata": metadata,
        "fits": fits,
    }


def run(args):
    if args.factors < 2:
        raise ValueError("at least two factors are required for shuffled-pair controls")
    if args.target_layer - args.source_layer < 2:
        raise ValueError("the exclusive layer range must contain at least one MLP layer")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    texts = read_texts(args.train_contexts, args.heldout_contexts)
    result = {
        "description": "Retrospective intermediate-MLP participation-ratio test",
        "source_layer": args.source_layer,
        "target_layer": args.target_layer,
        "exclusive_intermediate_layers": list(range(args.source_layer + 1, args.target_layer)),
        "fit_seeds": args.fit_seeds,
        "factors": args.factors,
        "iterations": args.iterations,
        "permutations": args.permutations,
        "architectures": {},
    }
    for architecture in args.architectures:
        result["architectures"][architecture] = run_architecture(
            args, architecture, tokenizer, texts,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architectures", choices=REPOSITORIES, nargs="+", default=tuple(REPOSITORIES))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--source-layer", type=int, default=8)
    parser.add_argument("--target-layer", type=int, default=12)
    parser.add_argument("--fit-seeds", type=int, nargs="+", default=(0, 1, 2, 3))
    parser.add_argument("--factors", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--factor-batch", type=int, default=8)
    parser.add_argument("--pair-batch", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--target-positions", type=int, default=3)
    parser.add_argument("--train-contexts", type=int, default=4)
    parser.add_argument("--heldout-contexts", type=int, default=8)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--null-seed", type=int, default=20000)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "intermediate_mlp_pr_results.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
