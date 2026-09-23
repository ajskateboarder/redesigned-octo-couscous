import sys
import os
from pathlib import Path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(current_dir))

import argparse
import json

import torch
from torch import nn, vmap
from torch.func import grad_and_value
from scipy.optimize import linear_sum_assignment
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import GPT2Tokenizer

from dct import LinearDCT, directional_hessian_outputs
from intermediate_mlp_pr_experiment import (
    MiddleSpanOperator,
    REPOSITORIES,
    SpanDelta,
    collect_middle_inputs,
    participation_ratio,
    read_texts,
)
from tensor_model import load_tensor_gpt


ROOT = Path(__file__).resolve().parent


class InclusiveMLPFeatures(nn.Module):
    """Return pre-down-projection MLP features for every block from s through t."""

    def __init__(self, model, state, source_layer, target_layer, target_positions):
        super().__init__()
        if not 0 <= source_layer <= target_layer < model.config.n_layer:
            raise ValueError("inclusive feature span must satisfy 0 <= source <= target < n_layer")
        self.blocks = model.transformer.h[source_layer:target_layer + 1]
        self.source_layer = source_layer
        self.target_layer = target_layer
        self.target_positions = target_positions
        self.register_buffer("reference_values", state[0])
        self.register_buffer("initial_values", state[1])
        self.register_buffer("first_values", state[2])
        self.device = state[0].device
        self.dtype = state[0].dtype

    @property
    def included_layers(self):
        return list(range(self.source_layer, self.target_layer + 1))

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
        for block in self.blocks:
            values = block.lambdas[0] * values + block.lambdas[1] * initial_values
            attention, first_values = block.attn(
                F.rms_norm(values, (values.shape[-1],)), first_values,
            )
            values = values + attention
            mlp_input = F.rms_norm(values, (values.shape[-1],))
            features.append(
                self.hidden_features(block.mlp, mlp_input)[:, self.target_positions].mean(dim=1)
            )
            values = values + block.mlp(mlp_input)
        return torch.cat(features, dim=1)


class ParticipationRegularizedDCT:
    def __init__(self, num_factors, penalty_weight, penalty_scale):
        self.num_factors = num_factors
        self.penalty_weight = penalty_weight
        self.penalty_scale = penalty_scale

    def fit(
        self, delta, features, X, Y, batch_size=1, factor_batch_size=16,
        max_iters=5, beta=1.0, initial_factors=None,
    ):
        self.num_samples, _, self.d_source = X.shape
        self.device = delta.device
        if initial_factors is None:
            self.L = F.normalize(torch.randn(
                self.d_source, self.num_factors, device=self.device, dtype=torch.float32,
            ), dim=0)
            self.R = F.normalize(torch.randn(
                self.d_source, self.num_factors, device=self.device, dtype=torch.float32,
            ), dim=0)
            self.U = F.normalize(torch.randn(
                Y.shape[-1], self.num_factors, device=self.device, dtype=torch.float32,
            ), dim=0)
        else:
            initial_u, initial_left, initial_right = initial_factors
            self.U = initial_u.to(self.device).clone()
            self.L = initial_left.to(self.device).clone()
            self.R = initial_right.to(self.device).clone()
        feature_dimensions = features(
            torch.zeros(self.d_source, device=self.device), X[:1], Y[:1],
        ).shape[-1]
        self.objective_values = []
        self.score_energy_values = []
        self.normalized_pr_values = []

        def target_scores(u, left, right, context_values, context_clean):
            target_response = directional_hessian_outputs(
                delta, left, right, context_values, context_clean,
            )
            return target_response.float() @ u.float()

        def normalized_participation(left, right, context_values, context_clean):
            intermediate_response = directional_hessian_outputs(
                features, left, right, context_values, context_clean,
            )
            return participation_ratio(intermediate_response) / feature_dimensions

        def objective(current_u, current_left, current_right, context_values, context_clean):
            scores = target_scores(
                current_u, current_left, current_right, context_values, context_clean,
            )
            if self.penalty_weight == 0:
                normalized_pr = torch.zeros_like(scores)
            else:
                normalized_pr = normalized_participation(
                    current_left, current_right, context_values, context_clean,
                )
            penalty = self.penalty_weight * self.penalty_scale * normalized_pr
            return (0.5 * scores.square() - penalty).sum(), (scores, normalized_pr)

        objective_with_grad = grad_and_value(
            objective, argnums=(0, 1, 2), has_aux=True,
        )

        def gradients(u, left, right, context_values, context_clean):
            updates, (_, auxiliary) = objective_with_grad(
                u, left, right, context_values, context_clean,
            )
            scores, normalized_pr = auxiliary
            return scores.detach(), normalized_pr.detach(), updates

        batched_gradients = vmap(
            gradients, in_dims=(1, 1, 1, None, None),
            out_dims=(1, 1, (1, 1, 1)), chunk_size=factor_batch_size,
        )

        for _ in range(max_iters):
            with torch.no_grad():
                self.L, _ = torch.linalg.qr(self.L)
                self.R, _ = torch.linalg.qr(self.R)
            score_energy = torch.zeros(self.num_factors, device=self.device)
            normalized_pr_sum = torch.zeros(self.num_factors, device=self.device)
            update_u = torch.zeros_like(self.U)
            update_left = torch.zeros_like(self.L)
            update_right = torch.zeros_like(self.R)
            context_count = 0
            for start in range(0, self.num_samples, batch_size):
                context_values = X[start:start + batch_size].to(self.device)
                context_clean = Y[start:start + batch_size].to(self.device)
                scores, normalized_pr, updates = batched_gradients(
                    self.U, self.L, self.R, context_values, context_clean,
                )
                with torch.no_grad():
                    score_energy += scores.square().sum(dim=0)
                    normalized_pr_sum += normalized_pr.sum(dim=0)
                    update_u += updates[0]
                    update_left += updates[1]
                    update_right += updates[2]
                context_count += len(context_values)
            with torch.no_grad():
                update_u /= context_count
                update_left /= context_count
                update_right /= context_count
                self.U = F.normalize(beta * update_u + (1 - beta) * self.U, dim=0)
                self.L = F.normalize(beta * update_left + (1 - beta) * self.L, dim=0)
                self.R = F.normalize(beta * update_right + (1 - beta) * self.R, dim=0)
                mean_energy = score_energy / context_count
                mean_pr = normalized_pr_sum / context_count
                self.score_energy_values.append(float(mean_energy.sum()))
                self.normalized_pr_values.append(float(mean_pr.mean()))
                self.objective_values.append(float(
                    0.5 * mean_energy.sum()
                    - self.penalty_weight * self.penalty_scale * mean_pr.sum()
                ))
        return self.U, self.L, self.R


def make_operators(model, state, args):
    span = MiddleSpanOperator(
        model, state, args.source_layer, args.target_layer,
    ).to(state[0].device)
    clean = span(state[0]).detach()
    delta = SpanDelta(span, slice(-args.target_positions, None))
    features = InclusiveMLPFeatures(
        model, state, args.source_layer, args.target_layer,
        slice(-args.target_positions, None),
    )
    with torch.no_grad():
        features(torch.zeros(state[0].shape[-1], device=state[0].device), state[0][:1], clean[:1])
    return clean, delta, features


def evaluate(dictionary, delta, features, values, clean, context_batch_size=8):
    energies = []
    participation_ratios = []

    def factor_statistics(u, left, right, context_values, context_clean):
        target_response = directional_hessian_outputs(
            delta, left, right, context_values, context_clean,
        )
        intermediate_response = directional_hessian_outputs(
            features, left, right, context_values, context_clean,
        )
        scores = target_response.float() @ u.float()
        return scores.square(), participation_ratio(intermediate_response)

    batched_statistics = vmap(
        factor_statistics, in_dims=(1, 1, 1, None, None), out_dims=(1, 1),
        chunk_size=dictionary.num_factors,
    )
    with sdpa_kernel(SDPBackend.MATH):
        for start in range(0, len(values), context_batch_size):
            batch_energy, batch_pr = batched_statistics(
                dictionary.U, dictionary.L, dictionary.R,
                values[start:start + context_batch_size],
                clean[start:start + context_batch_size],
            )
            energies.append(batch_energy)
            participation_ratios.append(batch_pr)
    energies = torch.cat(energies)
    participation_ratios = torch.cat(participation_ratios)
    return {
        "factor_context_energy": energies.detach().cpu().tolist(),
        "mean_total_energy": float(energies.sum(dim=1).mean()),
        "median_participation_ratio": float(participation_ratios.median()),
        "mean_participation_ratio": float(participation_ratios.mean()),
    }


def initial_factors(initialization, delta, values, clean, args, seed):
    torch.manual_seed(seed)
    if initialization == "random":
        dimensions = values.shape[-1]
        initial_left = F.normalize(
            torch.randn(dimensions, args.factors, device=values.device), dim=0,
        )
        initial_right = F.normalize(
            torch.randn(dimensions, args.factors, device=values.device), dim=0,
        )
        initial_u = F.normalize(
            torch.randn(dimensions, args.factors, device=values.device), dim=0,
        )
        return initial_u, initial_left, initial_right
    if initialization == "jacobian":
        linear = LinearDCT(num_factors=args.factors)
        initial_u, initial_v = linear.fit(
            delta, values, clean, method="projected", batch_size=1,
            dim_output_projection=args.jacobian_projection,
            factor_batch_size=args.factor_batch,
        )
        initial_v = F.normalize(initial_v.float(), dim=0)
        return F.normalize(initial_u.float(), dim=0), initial_v, initial_v.clone()
    raise ValueError(f"unknown initialization: {initialization}")


def factor_span(vector_left, vector_right, tolerance=1e-4):
    vectors = torch.stack((vector_left.float(), vector_right.float()), dim=1)
    basis, singular_values, _ = torch.linalg.svd(vectors, full_matrices=False)
    rank = int((singular_values > singular_values.max() * tolerance).sum())
    return basis[:, :max(rank, 1)]


def cross_seed_alignment(reference, candidate):
    factors = reference.num_factors
    span_similarity = torch.empty(factors, factors)
    for reference_index in range(factors):
        reference_span = factor_span(
            reference.L[:, reference_index], reference.R[:, reference_index],
        )
        for candidate_index in range(factors):
            candidate_span = factor_span(
                candidate.L[:, candidate_index], candidate.R[:, candidate_index],
            )
            canonical = torch.linalg.svdvals(reference_span.T @ candidate_span)
            span_similarity[reference_index, candidate_index] = (
                canonical.square().sum() / max(reference_span.shape[1], candidate_span.shape[1])
            )
    output_similarity = (reference.U.float().T @ candidate.U.float()).abs().cpu()
    joint_similarity = span_similarity * output_similarity
    rows, columns = linear_sum_assignment(joint_similarity.numpy(), maximize=True)
    matched_span = span_similarity[rows, columns]
    matched_joint = joint_similarity[rows, columns]
    return {
        "reference_factor_indices": rows.tolist(),
        "candidate_factor_indices": columns.tolist(),
        "matched_span_similarity": matched_span.tolist(),
        "matched_joint_similarity": matched_joint.tolist(),
        "median_matched_span_similarity": float(matched_span.median()),
        "median_matched_joint_similarity": float(matched_joint.median()),
        "max_matched_joint_similarity": float(matched_joint.max()),
        "matched_joint_above_0_8": int((matched_joint > 0.8).sum()),
    }


def run_architecture(args, architecture, tokenizer, texts):
    model, _, metadata = load_tensor_gpt(REPOSITORIES[architecture], args.device)
    model.requires_grad_(False)
    states = [
        collect_middle_inputs(model, tokenizer, split, args.source_layer, args.sequence_length)
        for split in texts
    ]
    train_clean, train_delta, train_features = make_operators(model, states[0], args)
    heldout_clean, heldout_delta, heldout_features = make_operators(model, states[1], args)
    fits = []
    fitted_dictionaries = {}
    for initialization in args.initializations:
        for seed in args.fit_seeds:
            initialization_values = initial_factors(
                initialization, train_delta, states[0][0], train_clean, args, seed,
            )
            baseline = ParticipationRegularizedDCT(args.factors, 0.0, 1.0)
            with sdpa_kernel(SDPBackend.MATH):
                baseline.fit(
                    train_delta, train_features, states[0][0], train_clean,
                    batch_size=args.context_batch, factor_batch_size=args.factor_batch,
                    max_iters=args.iterations, initial_factors=initialization_values,
                )
            baseline_train = evaluate(
                baseline, train_delta, train_features, states[0][0], train_clean,
                args.context_batch,
            )
            penalty_scale = baseline_train["mean_total_energy"] / args.factors
            sweep = []
            for penalty_weight in args.penalty_weights:
                if penalty_weight == 0:
                    dictionary = baseline
                else:
                    dictionary = ParticipationRegularizedDCT(
                        args.factors, penalty_weight, penalty_scale,
                    )
                    with sdpa_kernel(SDPBackend.MATH):
                        dictionary.fit(
                            train_delta, train_features, states[0][0], train_clean,
                            batch_size=args.context_batch, factor_batch_size=args.factor_batch,
                            max_iters=args.iterations, initial_factors=initialization_values,
                        )
                fitted_dictionaries[(initialization, seed, penalty_weight)] = dictionary
                sweep.append({
                    "penalty_weight": penalty_weight,
                    "penalty_scale": penalty_scale,
                    "objective_trace": dictionary.objective_values,
                    "score_energy_trace": dictionary.score_energy_values,
                    "normalized_pr_trace": dictionary.normalized_pr_values,
                    "train": baseline_train if penalty_weight == 0 else evaluate(
                        dictionary, train_delta, train_features, states[0][0], train_clean,
                        args.context_batch,
                    ),
                    "heldout": evaluate(
                        dictionary, heldout_delta, heldout_features,
                        states[1][0], heldout_clean, args.context_batch,
                    ),
                    "final_cosine_left_right": (
                        F.cosine_similarity(dictionary.L, dictionary.R, dim=0).detach().cpu().tolist()
                    ),
                })
            fits.append({"initialization": initialization, "fit_seed": seed, "sweep": sweep})

    stability = []
    for initialization in args.initializations:
        for penalty_weight in args.penalty_weights:
            for left_index, left_seed in enumerate(args.fit_seeds):
                for right_seed in args.fit_seeds[left_index + 1:]:
                    stability.append({
                        "initialization": initialization,
                        "penalty_weight": penalty_weight,
                        "left_seed": left_seed,
                        "right_seed": right_seed,
                        **cross_seed_alignment(
                            fitted_dictionaries[(initialization, left_seed, penalty_weight)],
                            fitted_dictionaries[(initialization, right_seed, penalty_weight)],
                        ),
                    })
    return {
        "repository": REPOSITORIES[architecture],
        "model_metadata": metadata,
        "fits": fits,
        "cross_seed_stability": stability,
    }


def run(args):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    texts = read_texts(args.train_contexts, args.heldout_contexts)
    result = {
        "description": "Direct source-through-target MLP PR-penalty sweep",
        "source_layer": args.source_layer,
        "target_layer": args.target_layer,
        "included_pr_layers": list(range(args.source_layer, args.target_layer + 1)),
        "initializations": args.initializations,
        "fit_seeds": args.fit_seeds,
        "factors": args.factors,
        "iterations": args.iterations,
        "factor_batch": args.factor_batch,
        "context_batch": args.context_batch,
        "jacobian_projection": args.jacobian_projection,
        "train_contexts": args.train_contexts,
        "heldout_contexts": args.heldout_contexts,
        "penalty_weights": args.penalty_weights,
        "architectures": {},
    }
    for architecture in args.architectures:
        result["architectures"][architecture] = run_architecture(
            args, architecture, tokenizer, texts,
        )
        args.output.write_text(json.dumps(result, indent=2))
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architectures", choices=REPOSITORIES, nargs="+", default=("bilinear",))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--source-layer", type=int, default=8)
    parser.add_argument("--target-layer", type=int, default=12)
    parser.add_argument("--fit-seeds", type=int, nargs="+", default=(0, 1))
    parser.add_argument(
        "--initializations", choices=("random", "jacobian"), nargs="+",
        default=("random", "jacobian"),
    )
    parser.add_argument("--factors", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--factor-batch", type=int, default=4)
    parser.add_argument("--context-batch", type=int, default=8)
    parser.add_argument("--jacobian-projection", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--target-positions", type=int, default=3)
    parser.add_argument("--train-contexts", type=int, default=4)
    parser.add_argument("--heldout-contexts", type=int, default=8)
    parser.add_argument("--penalty-weights", type=float, nargs="+", default=(0.0, 1.0, 10.0, 100.0))
    parser.add_argument("--output", type=Path, default=ROOT / "pr_penalty_inclusive_results.json")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())