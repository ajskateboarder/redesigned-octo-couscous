import argparse
import json
from pathlib import Path

import torch
from torch import nn, vmap
from torch.func import grad_and_value
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import GPT2Tokenizer

from bilinear_quadratic_experiment import (
    BilinearDelta,
    BilinearSpan,
    ROOT,
    cross_seed_alignment,
    evaluate,
    ordered_cross_hessian_outputs,
    random_swap_check,
    read_ood_texts,
    source_weight_alignment,
)
from intermediate_mlp_pr_experiment import (
    REPOSITORIES,
    collect_middle_inputs,
    participation_ratio,
    read_texts,
)
from tensor_model import load_tensor_gpt


class BilinearIntermediateMLPFeatures(nn.Module):
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

    def forward(self, theta_left, theta_right, values, clean):
        indices = self.context_indices(values)
        initial_values = self.initial_values[indices]
        first_values = self.first_values[indices]
        features = []
        for relative_layer, block in enumerate(self.blocks):
            values = block.lambdas[0] * values + block.lambdas[1] * initial_values
            attention, first_values = block.attn(
                F.rms_norm(values, (values.shape[-1],)), first_values,
            )
            values = values + attention
            mlp_input = F.rms_norm(values, (values.shape[-1],))
            if relative_layer == 0:
                left = block.mlp.left(mlp_input + theta_left)
                if block.mlp.config.gated:
                    left = F.silu(left)
                hidden = left * block.mlp.right(mlp_input + theta_right)
                mlp_output = block.mlp.Down(hidden) + block.mlp.Down_bias
            else:
                left = block.mlp.left(mlp_input)
                if block.mlp.config.gated:
                    left = F.silu(left)
                hidden = left * block.mlp.right(mlp_input)
                mlp_output = block.mlp.Down(hidden) + block.mlp.Down_bias
                features.append(hidden[:, self.target_positions].mean(dim=1))
            values = values + mlp_output
        if not features:
            raise ValueError("the exclusive layer range contains no MLP layers")
        return torch.cat(features, dim=1)


class ParticipationRegularizedBilinearDCT:
    def __init__(self, num_factors, penalty_weight, penalty_scale):
        self.num_factors = num_factors
        self.penalty_weight = penalty_weight
        self.penalty_scale = penalty_scale

    def fit(self, delta, features, values, clean, batch_size, factor_batch_size, max_iters, seed):
        torch.manual_seed(seed)
        dimensions = values.shape[-1]
        self.L = F.normalize(torch.randn(dimensions, self.num_factors, device=values.device), dim=0)
        self.R = F.normalize(torch.randn(dimensions, self.num_factors, device=values.device), dim=0)
        self.U = F.normalize(torch.randn(clean.shape[-1], self.num_factors, device=values.device), dim=0)
        feature_dimensions = features(
            torch.zeros(dimensions, device=values.device),
            torch.zeros(dimensions, device=values.device),
            values[:1], clean[:1],
        ).shape[-1]
        self.score_energy_trace = []
        self.normalized_pr_trace = []
        self.objective_trace = []

        def statistics(u, left, right, context_values, context_clean):
            target_response = ordered_cross_hessian_outputs(
                delta, left, right, context_values, context_clean,
            )
            feature_response = ordered_cross_hessian_outputs(
                features, left, right, context_values, context_clean,
            )
            scores = target_response.float() @ u.float()
            normalized_pr = participation_ratio(feature_response) / feature_dimensions
            return scores, normalized_pr

        def objective(u, left, right, context_values, context_clean):
            scores, normalized_pr = statistics(
                u, left, right, context_values, context_clean,
            )
            value = (
                0.5 * scores.square().sum()
                - self.penalty_weight * self.penalty_scale * normalized_pr.sum()
            )
            return value, (scores, normalized_pr)

        objective_with_grad = grad_and_value(
            objective, argnums=(0, 1, 2), has_aux=True,
        )

        def gradients(u, left, right, context_values, context_clean):
            updates, (_, statistics_value) = objective_with_grad(
                u, left, right, context_values, context_clean,
            )
            return statistics_value[0].detach(), statistics_value[1].detach(), updates

        batched_gradients = vmap(
            gradients, in_dims=(1, 1, 1, None, None),
            out_dims=(1, 1, (1, 1, 1)), chunk_size=factor_batch_size,
        )

        for _ in range(max_iters):
            with torch.no_grad():
                self.L, _ = torch.linalg.qr(self.L)
                self.R, _ = torch.linalg.qr(self.R)
            score_energy = torch.zeros(self.num_factors, device=values.device)
            normalized_pr_sum = torch.zeros(self.num_factors, device=values.device)
            update_u = torch.zeros_like(self.U)
            update_left = torch.zeros_like(self.L)
            update_right = torch.zeros_like(self.R)
            context_count = 0
            for start in range(0, len(values), batch_size):
                batch_values = values[start:start + batch_size]
                batch_clean = clean[start:start + batch_size]
                scores, normalized_pr, updates = batched_gradients(
                    self.U, self.L, self.R, batch_values, batch_clean,
                )
                with torch.no_grad():
                    score_energy += scores.square().sum(dim=0)
                    normalized_pr_sum += normalized_pr.sum(dim=0)
                    update_u += updates[0]
                    update_left += updates[1]
                    update_right += updates[2]
                context_count += len(batch_values)
            with torch.no_grad():
                self.U = F.normalize(update_u / context_count, dim=0)
                self.L = F.normalize(update_left / context_count, dim=0)
                self.R = F.normalize(update_right / context_count, dim=0)
                mean_energy = score_energy / context_count
                mean_pr = normalized_pr_sum / context_count
                self.score_energy_trace.append(float(mean_energy.sum()))
                self.normalized_pr_trace.append(float(mean_pr.mean()))
                self.objective_trace.append(float(
                    0.5 * mean_energy.sum()
                    - self.penalty_weight * self.penalty_scale * mean_pr.sum()
                ))
        return self.U, self.L, self.R


def make_operators(model, state, args):
    operator = BilinearSpan(model, state, args.source_layer, args.target_layer).to(state[0].device)
    zero = torch.zeros(state[0].shape[-1], device=state[0].device)
    with torch.no_grad():
        clean = operator(zero, zero, state[0]).detach()
    delta = BilinearDelta(operator, slice(-args.target_positions, None))
    features = BilinearIntermediateMLPFeatures(
        model, state, args.source_layer, args.target_layer,
        slice(-args.target_positions, None),
    )
    return clean, delta, features


def evaluate_pr(dictionary, features, values, clean, context_batch_size):
    def factor_pr(left, right, context_values, context_clean):
        response = ordered_cross_hessian_outputs(
            features, left, right, context_values, context_clean,
        )
        return participation_ratio(response)

    batched_pr = vmap(
        factor_pr, in_dims=(1, 1, None, None), out_dims=1,
        chunk_size=dictionary.num_factors,
    )
    values_by_context = []
    with sdpa_kernel(SDPBackend.MATH):
        for start in range(0, len(values), context_batch_size):
            values_by_context.append(batched_pr(
                dictionary.L, dictionary.R,
                values[start:start + context_batch_size],
                clean[start:start + context_batch_size],
            ))
    values_by_context = torch.cat(values_by_context)
    return {
        "mean_participation_ratio": float(values_by_context.mean()),
        "median_participation_ratio": float(values_by_context.median()),
        "factor_mean_participation_ratio": values_by_context.mean(dim=0).detach().cpu().tolist(),
    }


def factor_diversity(dictionary):
    factors = dictionary.num_factors
    if factors < 2:
        return {"pair_count": 0}
    pair_indices = torch.triu_indices(factors, factors, offset=1)
    result = {"pair_count": pair_indices.shape[1]}
    pair_similarities = {}
    for name, matrix in (("u", dictionary.U), ("left", dictionary.L), ("right", dictionary.R)):
        gram = (matrix.float().T @ matrix.float()).abs()
        values = gram[pair_indices.unbind()]
        pair_similarities[name] = values
        result[f"median_{name}_absolute_cosine"] = float(values.median())
        result[f"maximum_{name}_absolute_cosine"] = float(values.max())
    joint = pair_similarities["u"] * pair_similarities["left"] * pair_similarities["right"]
    result["median_joint_similarity"] = float(joint.median())
    result["maximum_joint_similarity"] = float(joint.max())
    return result


def add_source_unit_diversity(alignment):
    units = torch.tensor(alignment["best_source_unit_indices"])
    _, counts = units.unique(return_counts=True)
    alignment["unique_best_source_units"] = int(len(counts))
    alignment["unique_best_source_unit_fraction"] = float(len(counts) / len(units))
    alignment["maximum_best_source_unit_multiplicity"] = int(counts.max())
    return alignment


def source_alignment(dictionary, model, state, source_layer):
    source_mlp = model.transformer.h[source_layer].mlp
    alignment = source_weight_alignment(dictionary, source_mlp)
    if not source_mlp.config.gated:
        alignment["source_hidden_energy_basis"] = "ungated_branch_mixed_derivative"
        return alignment

    block = model.transformer.h[source_layer]
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
        values = block.lambdas[0] * state[0] + block.lambdas[1] * state[1]
        attention, _ = block.attn(
            F.rms_norm(values, (values.shape[-1],)), state[2],
        )
        mlp_input = F.rms_norm(values + attention, (values.shape[-1],))
        left_pre_activation = source_mlp.left(mlp_input).float()
        sigmoid = left_pre_activation.sigmoid()
        silu_derivative = sigmoid * (
            1 + left_pre_activation * (1 - sigmoid)
        )
        left_direction = source_mlp.Left.weight.float() @ dictionary.L.float()
        right_direction = source_mlp.Right.weight.float() @ dictionary.R.float()
        gate_energy = silu_derivative.square().sum(dim=(0, 1))
        energy = gate_energy[:, None] * (left_direction * right_direction).square()
        top_fraction = energy.max(dim=0).values / energy.sum(dim=0).clamp_min(1e-30)
    alignment["source_hidden_top1_energy_fraction"] = top_fraction.cpu().tolist()
    alignment["source_hidden_energy_basis"] = "contextual_silu_mixed_derivative"
    return alignment


def split_metrics(dictionary, delta, features, values, clean, context_batch_size):
    return {
        **evaluate(dictionary, delta, values, clean, context_batch_size),
        **evaluate_pr(dictionary, features, values, clean, context_batch_size),
    }


def run(args):
    if args.output is None:
        args.output = ROOT / f"{args.architecture}_quadratic_pr_results.json"
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    train_texts, heldout_texts = read_texts(args.train_contexts, args.heldout_contexts)
    ood_texts = read_ood_texts(args.ood_contexts)
    model, _, metadata = load_tensor_gpt(REPOSITORIES[args.architecture], args.device)
    model.requires_grad_(False)
    states = [
        collect_middle_inputs(model, tokenizer, texts, args.source_layer, args.sequence_length)
        for texts in (train_texts, heldout_texts, ood_texts)
    ]
    operators = [make_operators(model, state, args) for state in states]
    random_asymmetry = random_swap_check(
        operators[0][1], states[0][0][:args.context_batch],
        operators[0][0][:args.context_batch], args.factors, args.fit_seeds[0],
    )
    fits = []
    dictionaries = {}
    for seed in args.fit_seeds:
        seed_sweep = []
        baseline = ParticipationRegularizedBilinearDCT(args.factors, 0.0, 1.0)
        with sdpa_kernel(SDPBackend.MATH):
            baseline.fit(
                operators[0][1], operators[0][2], states[0][0], operators[0][0],
                args.fit_context_batch, args.factor_batch, args.iterations, seed,
            )
        baseline_train = split_metrics(
            baseline, operators[0][1], operators[0][2],
            states[0][0], operators[0][0], args.context_batch,
        )
        penalty_scale = (
            args.penalty_scale
            if args.penalty_scale is not None
            else baseline_train["mean_total_energy"] / args.factors
        )
        for penalty_weight in args.penalty_weights:
            if penalty_weight == 0:
                dictionary = baseline
            else:
                dictionary = ParticipationRegularizedBilinearDCT(
                    args.factors, penalty_weight, penalty_scale,
                )
                with sdpa_kernel(SDPBackend.MATH):
                    dictionary.fit(
                        operators[0][1], operators[0][2], states[0][0], operators[0][0],
                        args.fit_context_batch, args.factor_batch, args.iterations, seed,
                    )
            dictionaries[(seed, penalty_weight)] = dictionary
            split_results = {}
            for split_name, state, (clean, delta, features) in zip(
                ("train", "heldout", "ood_wikitext"), states, operators,
            ):
                if split_name == "train" and penalty_weight == 0:
                    split_results[split_name] = baseline_train
                else:
                    split_results[split_name] = split_metrics(
                        dictionary, delta, features, state[0], clean, args.context_batch,
                    )
            split_results["ood_wikitext"]["energy_retention_vs_heldout"] = (
                split_results["ood_wikitext"]["mean_total_energy"]
                / split_results["heldout"]["mean_total_energy"]
            )
            split_results["heldout"]["energy_retention_vs_train"] = (
                split_results["heldout"]["mean_total_energy"]
                / split_results["train"]["mean_total_energy"]
            )
            split_results["ood_wikitext"]["energy_retention_vs_train"] = (
                split_results["ood_wikitext"]["mean_total_energy"]
                / split_results["train"]["mean_total_energy"]
            )
            seed_sweep.append({
                "penalty_weight": penalty_weight,
                "penalty_scale": penalty_scale,
                "score_energy_trace": dictionary.score_energy_trace,
                "normalized_pr_trace": dictionary.normalized_pr_trace,
                "objective_trace": dictionary.objective_trace,
                **split_results,
                "factor_diversity": factor_diversity(dictionary),
                "source_weight_alignment": add_source_unit_diversity(
                    source_alignment(
                        dictionary, model, states[0], args.source_layer,
                    )
                ),
            })
        fits.append({"fit_seed": seed, "sweep": seed_sweep})
        args.output.write_text(json.dumps({"fits": fits}, indent=2))

    stability = {}
    for penalty_weight in args.penalty_weights:
        comparisons = []
        for left_index, left_seed in enumerate(args.fit_seeds):
            for right_seed in args.fit_seeds[left_index + 1:]:
                comparisons.append({
                    "left_seed": left_seed,
                    "right_seed": right_seed,
                    **cross_seed_alignment(
                        dictionaries[(left_seed, penalty_weight)],
                        dictionaries[(right_seed, penalty_weight)],
                    ),
                })
        stability[str(penalty_weight)] = comparisons
    result = {
        "description": "Branch-intervention DCT with downstream MLP PR penalty",
        "architecture": args.architecture,
        "repository": REPOSITORIES[args.architecture],
        "model_metadata": metadata,
        "source_layer": args.source_layer,
        "target_layer": args.target_layer,
        "intermediate_layers": operators[0][2].intermediate_layers,
        "fit_seeds": args.fit_seeds,
        "factors": args.factors,
        "iterations": args.iterations,
        "penalty_weights": args.penalty_weights,
        "penalty_scale_override": args.penalty_scale,
        "train_contexts": args.train_contexts,
        "heldout_contexts": args.heldout_contexts,
        "ood_dataset": "Salesforce/wikitext:wikitext-2-raw-v1:test",
        "ood_contexts": args.ood_contexts,
        "random_direction_swap_check": random_asymmetry,
        "fits": fits,
        "cross_seed_stability_by_penalty": stability,
    }
    args.output.write_text(json.dumps(result, indent=2))
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=("bilinear", "swiglu"), default="bilinear")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--source-layer", type=int, default=8)
    parser.add_argument("--target-layer", type=int, default=12)
    parser.add_argument("--fit-seeds", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument("--factors", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--factor-batch", type=int, default=8)
    parser.add_argument("--fit-context-batch", type=int, default=1)
    parser.add_argument("--context-batch", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--target-positions", type=int, default=3)
    parser.add_argument("--train-contexts", type=int, default=32)
    parser.add_argument("--heldout-contexts", type=int, default=64)
    parser.add_argument("--ood-contexts", type=int, default=64)
    parser.add_argument("--penalty-weights", type=float, nargs="+", default=(0.0, 0.01, 0.03, 0.1))
    parser.add_argument("--penalty-scale", type=float)
    parser.add_argument(
        "--output", type=Path,
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())