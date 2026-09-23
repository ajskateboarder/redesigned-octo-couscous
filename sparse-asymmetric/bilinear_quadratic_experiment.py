import argparse
import json
import os
import sys
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(current_dir))

import torch
from datasets import load_dataset
from scipy.optimize import linear_sum_assignment
from torch import nn, vmap
from torch.func import grad_and_value, jvp
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import GPT2Tokenizer

from intermediate_mlp_pr_experiment import REPOSITORIES, collect_middle_inputs, read_texts
from tensor_model import load_tensor_gpt


ROOT = Path(__file__).resolve().parent


def read_ood_texts(count):
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    texts = [
        text.strip() for text in dataset["text"]
        if text.strip() and not text.strip().startswith("=")
    ]
    if count > len(texts):
        raise ValueError("requested more OOD texts than are available")
    return texts[:count]


def ordered_cross_hessian_outputs(function, left, right, values, clean):
    zero_left = torch.zeros_like(left)
    zero_right = torch.zeros_like(right)

    def left_direction_at(current_right):
        return jvp(
            lambda current_left: function(
                current_left, current_right, values, clean,
            ),
            (zero_left,), (left,),
        )[1]

    return jvp(left_direction_at, (zero_right,), (right,))[1]


class BilinearSpan(nn.Module):
    def __init__(self, model, state, source_layer, target_layer):
        super().__init__()
        if not 0 <= source_layer < target_layer <= model.config.n_layer:
            raise ValueError("span must satisfy 0 <= source < target <= n_layer")
        source_mlp = model.transformer.h[source_layer].mlp
        if not hasattr(source_mlp, "Left") or source_mlp.config.gated:
            raise ValueError("ordered branch interventions require an ungated bilinear source MLP")
        self.blocks = model.transformer.h[source_layer:target_layer]
        self.register_buffer("reference_values", state[0])
        self.register_buffer("initial_values", state[1])
        self.register_buffer("first_values", state[2])
        self.device = state[0].device
        self.dtype = state[0].dtype

    def context_indices(self, values):
        distances = (self.reference_values[:, None] - values[None, :]).square().flatten(2).sum(2)
        return distances.argmin(dim=0)

    def forward(self, theta_left, theta_right, values):
        indices = self.context_indices(values)
        initial_values = self.initial_values[indices]
        first_values = self.first_values[indices]
        for relative_layer, block in enumerate(self.blocks):
            values = block.lambdas[0] * values + block.lambdas[1] * initial_values
            attention, first_values = block.attn(
                F.rms_norm(values, (values.shape[-1],)), first_values,
            )
            values = values + attention
            mlp_input = F.rms_norm(values, (values.shape[-1],))
            if relative_layer == 0:
                hidden = (
                    block.mlp.Left(mlp_input + theta_left)
                    * block.mlp.Right(mlp_input + theta_right)
                )
                mlp_output = block.mlp.Down(hidden) + block.mlp.Down_bias
            else:
                mlp_output = block.mlp(mlp_input)
            values = values + mlp_output
        return values


class BilinearDelta(nn.Module):
    def __init__(self, operator, target_positions):
        super().__init__()
        self.operator = operator
        self.device = operator.device
        self.dtype = operator.dtype
        self.target_positions = target_positions

    def forward(self, theta_left, theta_right, values, clean):
        output = self.operator(theta_left, theta_right, values)
        return (output - clean)[:, self.target_positions].mean(dim=1)


class BilinearQuadraticDCT:
    def __init__(self, num_factors):
        self.num_factors = num_factors

    def fit(self, delta, values, clean, batch_size, factor_batch_size, max_iters, seed):
        torch.manual_seed(seed)
        dimensions = values.shape[-1]
        self.L = F.normalize(torch.randn(
            dimensions, self.num_factors, device=values.device,
        ), dim=0)
        self.R = F.normalize(torch.randn(
            dimensions, self.num_factors, device=values.device,
        ), dim=0)
        self.U = F.normalize(torch.randn(
            clean.shape[-1], self.num_factors, device=values.device,
        ), dim=0)
        self.score_energy_trace = []

        def scores(u, left, right, context_values, context_clean):
            response = ordered_cross_hessian_outputs(
                delta, left, right, context_values, context_clean,
            )
            return response.float() @ u.float()

        def objective(u, left, right, context_values, context_clean):
            current_scores = scores(u, left, right, context_values, context_clean)
            return 0.5 * current_scores.square().sum(), current_scores

        objective_with_grad = grad_and_value(
            objective, argnums=(0, 1, 2), has_aux=True,
        )

        def gradients(u, left, right, context_values, context_clean):
            updates, (_, current_scores) = objective_with_grad(
                u, left, right, context_values, context_clean,
            )
            return current_scores.detach(), updates

        batched_gradients = vmap(
            gradients, in_dims=(1, 1, 1, None, None),
            out_dims=(1, (1, 1, 1)), chunk_size=factor_batch_size,
        )

        for _ in range(max_iters):
            with torch.no_grad():
                self.L, _ = torch.linalg.qr(self.L)
                self.R, _ = torch.linalg.qr(self.R)
            score_energy = torch.zeros(self.num_factors, device=values.device)
            update_u = torch.zeros_like(self.U)
            update_left = torch.zeros_like(self.L)
            update_right = torch.zeros_like(self.R)
            context_count = 0
            for start in range(0, len(values), batch_size):
                batch_values = values[start:start + batch_size]
                batch_clean = clean[start:start + batch_size]
                current_scores, updates = batched_gradients(
                    self.U, self.L, self.R, batch_values, batch_clean,
                )
                with torch.no_grad():
                    score_energy += current_scores.square().sum(dim=0)
                    update_u += updates[0]
                    update_left += updates[1]
                    update_right += updates[2]
                context_count += len(batch_values)
            with torch.no_grad():
                self.U = F.normalize(update_u / context_count, dim=0)
                self.L = F.normalize(update_left / context_count, dim=0)
                self.R = F.normalize(update_right / context_count, dim=0)
                self.score_energy_trace.append(float((score_energy / context_count).sum()))
        return self.U, self.L, self.R


def make_operators(model, state, args):
    operator = BilinearSpan(
        model, state, args.source_layer, args.target_layer,
    ).to(state[0].device)
    zero = torch.zeros(state[0].shape[-1], device=state[0].device)
    with torch.no_grad():
        clean = operator(zero, zero, state[0]).detach()
    delta = BilinearDelta(
        operator, slice(-args.target_positions, None),
    )
    return clean, delta


def evaluate(dictionary, delta, values, clean, context_batch_size):
    def factor_statistics(u, left, right, context_values, context_clean):
        direct = ordered_cross_hessian_outputs(
            delta, left, right, context_values, context_clean,
        )
        swapped = ordered_cross_hessian_outputs(
            delta, right, left, context_values, context_clean,
        )
        direct_scores = direct.float() @ u.float()
        swapped_scores = swapped.float() @ u.float()
        response_difference = (direct.float() - swapped.float()).norm(dim=-1)
        response_scale = direct.float().norm(dim=-1).clamp_min(1e-30)
        return direct_scores.square(), swapped_scores.square(), response_difference / response_scale

    batched_statistics = vmap(
        factor_statistics, in_dims=(1, 1, 1, None, None),
        out_dims=(1, 1, 1), chunk_size=dictionary.num_factors,
    )
    direct_energy = []
    swapped_energy = []
    response_asymmetry = []
    with sdpa_kernel(SDPBackend.MATH):
        for start in range(0, len(values), context_batch_size):
            metrics = batched_statistics(
                dictionary.U, dictionary.L, dictionary.R,
                values[start:start + context_batch_size],
                clean[start:start + context_batch_size],
            )
            direct_energy.append(metrics[0])
            swapped_energy.append(metrics[1])
            response_asymmetry.append(metrics[2])
    direct_energy = torch.cat(direct_energy)
    swapped_energy = torch.cat(swapped_energy)
    response_asymmetry = torch.cat(response_asymmetry)
    return {
        "mean_total_energy": float(direct_energy.sum(dim=1).mean()),
        "mean_swapped_total_energy": float(swapped_energy.sum(dim=1).mean()),
        "swapped_to_direct_energy_ratio": float(
            swapped_energy.sum(dim=1).mean() / direct_energy.sum(dim=1).mean().clamp_min(1e-30)
        ),
        "median_response_swap_asymmetry": float(response_asymmetry.median()),
        "factor_mean_energy": direct_energy.mean(dim=0).detach().cpu().tolist(),
    }


def random_swap_check(delta, values, clean, factors, seed):
    torch.manual_seed(seed)
    dimensions = values.shape[-1]
    left = F.normalize(torch.randn(dimensions, factors, device=values.device), dim=0)
    right = F.normalize(torch.randn(dimensions, factors, device=values.device), dim=0)

    def asymmetry(first, second):
        direct = ordered_cross_hessian_outputs(delta, first, second, values, clean)
        swapped = ordered_cross_hessian_outputs(delta, second, first, values, clean)
        return (direct.float() - swapped.float()).norm(dim=-1) / direct.float().norm(dim=-1).clamp_min(1e-30)

    with sdpa_kernel(SDPBackend.MATH):
        values_by_factor = vmap(
            asymmetry, in_dims=(1, 1), out_dims=1, chunk_size=factors,
        )(left, right)
    return {
        "median_relative_response_difference": float(values_by_factor.median()),
        "minimum_relative_response_difference": float(values_by_factor.min()),
        "maximum_relative_response_difference": float(values_by_factor.max()),
    }


def source_weight_alignment(dictionary, source_mlp):
    left_rows = F.normalize(source_mlp.Left.weight.float(), dim=1)
    right_rows = F.normalize(source_mlp.Right.weight.float(), dim=1)
    ordered = (dictionary.L.float().T @ left_rows.T).abs() * (
        dictionary.R.float().T @ right_rows.T
    ).abs()
    swapped = (dictionary.L.float().T @ right_rows.T).abs() * (
        dictionary.R.float().T @ left_rows.T
    ).abs()
    best_ordered, best_units = ordered.max(dim=1)
    best_swapped = swapped.max(dim=1).values
    local_response = (
        source_mlp.Left.weight.float() @ dictionary.L.float()
    ) * (source_mlp.Right.weight.float() @ dictionary.R.float())
    energy = local_response.square()
    top_fraction = energy.max(dim=0).values / energy.sum(dim=0).clamp_min(1e-30)
    return {
        "best_ordered_joint_weight_alignment": best_ordered.detach().cpu().tolist(),
        "best_swapped_joint_weight_alignment": best_swapped.detach().cpu().tolist(),
        "best_source_unit_indices": best_units.detach().cpu().tolist(),
        "source_hidden_top1_energy_fraction": top_fraction.detach().cpu().tolist(),
    }


def cross_seed_alignment(reference, candidate):
    direct = (
        (reference.U.float().T @ candidate.U.float()).abs()
        * (reference.L.float().T @ candidate.L.float()).abs()
        * (reference.R.float().T @ candidate.R.float()).abs()
    ).cpu()
    swapped = (
        (reference.U.float().T @ candidate.U.float()).abs()
        * (reference.L.float().T @ candidate.R.float()).abs()
        * (reference.R.float().T @ candidate.L.float()).abs()
    ).cpu()
    rows, columns = linear_sum_assignment(direct.numpy(), maximize=True)
    matched_direct = direct[rows, columns]
    matched_swapped = swapped[rows, columns]
    return {
        "left_seed_factor_indices": rows.tolist(),
        "right_seed_factor_indices": columns.tolist(),
        "matched_direct_similarity": matched_direct.tolist(),
        "matched_swapped_similarity": matched_swapped.tolist(),
        "median_direct_similarity": float(matched_direct.median()),
        "median_swapped_similarity": float(matched_swapped.median()),
        "maximum_direct_similarity": float(matched_direct.max()),
        "direct_matches_above_0_8": int((matched_direct > 0.8).sum()),
    }


def run(args):
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
    train_clean, train_delta = make_operators(model, states[0], args)
    heldout_clean, heldout_delta = make_operators(model, states[1], args)
    ood_clean, ood_delta = make_operators(model, states[2], args)
    random_asymmetry = random_swap_check(
        train_delta, states[0][0][:args.context_batch],
        train_clean[:args.context_batch], args.factors, args.fit_seeds[0],
    )
    dictionaries = {}
    fits = []
    for seed in args.fit_seeds:
        dictionary = BilinearQuadraticDCT(args.factors)
        with sdpa_kernel(SDPBackend.MATH):
            dictionary.fit(
                train_delta, states[0][0], train_clean,
                args.context_batch, args.factor_batch, args.iterations, seed,
            )
        dictionaries[seed] = dictionary
        heldout_metrics = evaluate(
            dictionary, heldout_delta, states[1][0], heldout_clean, args.context_batch,
        )
        ood_metrics = evaluate(
            dictionary, ood_delta, states[2][0], ood_clean, args.context_batch,
        )
        ood_metrics["energy_retention_vs_heldout"] = (
            ood_metrics["mean_total_energy"] / heldout_metrics["mean_total_energy"]
        )
        fits.append({
            "fit_seed": seed,
            "score_energy_trace": dictionary.score_energy_trace,
            "train": evaluate(
                dictionary, train_delta, states[0][0], train_clean, args.context_batch,
            ),
            "heldout": heldout_metrics,
            "ood_wikitext": ood_metrics,
            "source_weight_alignment": source_weight_alignment(
                dictionary, model.transformer.h[args.source_layer].mlp,
            ),
        })
    stability = []
    for left_index, left_seed in enumerate(args.fit_seeds):
        for right_seed in args.fit_seeds[left_index + 1:]:
            stability.append({
                "left_seed": left_seed,
                "right_seed": right_seed,
                **cross_seed_alignment(dictionaries[left_seed], dictionaries[right_seed]),
            })
    result = {
        "description": "Bilinear quadratic DCT with branch-specific cross-derivatives",
        "architecture": args.architecture,
        "repository": REPOSITORIES[args.architecture],
        "model_metadata": metadata,
        "source_layer": args.source_layer,
        "target_layer": args.target_layer,
        "fit_seeds": args.fit_seeds,
        "factors": args.factors,
        "iterations": args.iterations,
        "train_contexts": args.train_contexts,
        "heldout_contexts": args.heldout_contexts,
        "ood_dataset": "Salesforce/wikitext:wikitext-2-raw-v1:test",
        "ood_contexts": args.ood_contexts,
        "random_direction_swap_check": random_asymmetry,
        "fits": fits,
        "cross_seed_stability": stability,
    }
    args.output.write_text(json.dumps(result, indent=2))
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=("bilinear",), default="bilinear")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--source-layer", type=int, default=8)
    parser.add_argument("--target-layer", type=int, default=12)
    parser.add_argument("--fit-seeds", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument("--factors", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--factor-batch", type=int, default=8)
    parser.add_argument("--context-batch", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--target-positions", type=int, default=3)
    parser.add_argument("--train-contexts", type=int, default=32)
    parser.add_argument("--heldout-contexts", type=int, default=64)
    parser.add_argument("--ood-contexts", type=int, default=64)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "ordered_branch_results.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())