import argparse
import json
from pathlib import Path

from tqdm.auto import trange
import torch
from jaxtyping import Float, Int
from scipy.optimize import linear_sum_assignment
from torch import nn, vmap, Tensor
from torch.func import grad_and_value
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import GPT2Tokenizer

from bilinear_quadratic_experiment import (
    ROOT,
    evaluate,
    ordered_cross_hessian_outputs,
    random_swap_check,
    read_ood_texts,
    source_weight_alignment,
)
from bilinear_quadratic_pr_experiment import (
    add_source_unit_diversity,
    factor_diversity,
)
from intermediate_mlp_pr_experiment import (
    REPOSITORIES,
    collect_middle_inputs,
    read_texts,
)
from tensor_model import CausalSelfAttention, TensorGPT, apply_rotary_emb, load_tensor_gpt


class SlicedModel(nn.Module):
    def __init__(
        self,
        model: TensorGPT,
        state: tuple[
            Float[Tensor, "context sequence embedding"],
            Float[Tensor, "context sequence embedding"],
            Float[Tensor, "context sequence head head_dim"],
        ],
        source_layer: int,
        target_layer: int,
        target_positions: slice,
    ):
        super().__init__()
        if not 0 <= source_layer < target_layer <= model.config.n_layer:
            raise ValueError("span must satisfy 0 <= source < target <= n_layer")
        self.blocks = model.transformer.h[source_layer:target_layer]
        self.source_layer = source_layer
        self.target_positions = target_positions
        self.register_buffer("reference_values", state[0])
        self.register_buffer("initial_values", state[1])
        self.register_buffer("first_values", state[2])
        self.device = state[0].device
        self.dtype = state[0].dtype

    @property
    def attention_layers(self):
        return list(range(self.source_layer + 1, self.source_layer + len(self.blocks)))

    def context_indices(
        self,
        values: Float[Tensor, "batch sequence embedding"],
    ) -> Int[Tensor, "batch"]:
        distances = (self.reference_values[:, None] - values[None, :]).square().flatten(2).sum(2)
        return distances.argmin(dim=0)

    def attn(
        self,
        attention: CausalSelfAttention,
        values: Float[Tensor, "batch sequence embedding"],
        first_values: Float[Tensor, "batch sequence head head_dim"],
    ) -> tuple[
        Float[Tensor, "batch sequence embedding"],
        Float[Tensor, "batch sequence head head_dim"],
        Float[Tensor, "batch head position embedding"],
    ]:
        # squared attn
        if hasattr(attention, "c_q2"):
            raise NotImplementedError

        batch, sequence, dimensions = values.shape
        heads = attention.n_head
        head_dimensions = attention.head_dim
        query = attention.c_q(values).view(batch, sequence, heads, head_dimensions)
        key = attention.c_k(values).view(batch, sequence, heads, head_dimensions)
        current_value = attention.c_v(values).view(batch, sequence, heads, head_dimensions)
        if first_values is None:
            first_values = current_value
        value = (
            (1 - attention.lamb) * current_value
            + attention.lamb * first_values.view_as(current_value)
        )
        cosine, sine = attention.rotary(query)
        query = F.rms_norm(query, (head_dimensions,))
        key = F.rms_norm(key, (head_dimensions,))
        query = apply_rotary_emb(query, cosine, sine).transpose(1, 2)
        key = apply_rotary_emb(key, cosine, sine).transpose(1, 2)
        value_by_head = value.transpose(1, 2)
        logits = query @ key.transpose(-1, -2) / head_dimensions ** 0.5
        causal = torch.ones(sequence, sequence, device=values.device, dtype=torch.bool).tril()
        pattern = logits.masked_fill(~causal, float("-inf")).softmax(dim=-1)
        head_values = pattern @ value_by_head
        attention_output = attention.c_proj(
            head_values.transpose(1, 2).contiguous().view_as(values)
        )

        selected_heads = head_values[:, :, self.target_positions]
        output_weight = attention.c_proj.weight.view(dimensions, heads, head_dimensions)
        projected_heads = torch.einsum("bhpd,chd->bhpc", selected_heads, output_weight)
        return attention_output, first_values, projected_heads

    @staticmethod
    def mlp(mlp, values, theta_left=0.0, theta_right=0.0):
        left = mlp.left(values + theta_left)
        if mlp.config.gated:
            # fused F.silu has no nested forward-mode AD
            left = left * torch.sigmoid(left)
        return mlp.Down(left * mlp.right(values + theta_right)) + mlp.Down_bias

    def forward(
        self,
        theta_left: Float[Tensor, "embedding"],
        theta_right: Float[Tensor, "embedding"],
        values: Float[Tensor, "batch sequence embedding"],
        clean: Float[Tensor, "batch sequence embedding"]
        | Float[Tensor, "batch embedding"],
    ) -> tuple[Float[Tensor, "batch embedding"], Float[Tensor, "batch layer head position embedding"]]:
        indices = self.context_indices(values)
        initial_values = self.initial_values[indices]
        first_values = self.first_values[indices]
        ov_features = []

        for relative_layer, block in enumerate(self.blocks):
            values = block.lambdas[0] * values + block.lambdas[1] * initial_values
            attention_input = F.rms_norm(values, (values.shape[-1],))
            attention_output, first_values, ov = self.attn(block.attn, attention_input, first_values)
            values = values + attention_output
            mlp_input = F.rms_norm(values, (values.shape[-1],))
            if relative_layer == 0:
                mlp_output = self.mlp(block.mlp, mlp_input, theta_left, theta_right)
            else:
                ov_features.append(ov)
                mlp_output = self.mlp(block.mlp, mlp_input)
            values = values + mlp_output
        assert ov_features, "the exclusive layer range contains no attention layers"

        target = values[:, self.target_positions].mean(dim=1)
        if clean.ndim == 3:
            clean = clean[:, self.target_positions].mean(dim=1)
        target = target - clean        
        return target, torch.stack(ov_features, dim=1)


class AttentionTargetDelta(nn.Module):
    def __init__(self, operator):
        super().__init__()
        self.operator = operator
        self.device = operator.device
        self.dtype = operator.dtype

    def forward(self, theta_left, theta_right, values, clean):
        return self.operator(theta_left, theta_right, values, clean)[0]


def normalized_head_pr(response):
    energy = response.float().flatten(3).square().sum(dim=-1)
    pr = energy.sum(dim=-1).square() / energy.square().sum(dim=-1).clamp_min(1e-30)
    return pr / response.shape[2]


class AttentionRegularizedBilinearDCT:
    def __init__(
        self, num_factors, penalty_weight, penalty_scale,
        target_head_pr=None, target_penalty_mode="energy_weighted",
        penalty_warmup_iters=0, update_rate=1.0,
    ):
        if target_head_pr is not None and not 0.0 <= target_head_pr <= 1.0:
            raise ValueError("target head PR must be between zero and one")
        if target_penalty_mode not in {"energy_weighted", "unweighted"}:
            raise ValueError(f"unknown target penalty mode: {target_penalty_mode}")
        if penalty_warmup_iters < 0:
            raise ValueError("penalty warmup iterations must be nonnegative")
        if not 0.0 < update_rate <= 1.0:
            raise ValueError("update rate must be between zero and one")
        self.num_factors = num_factors
        self.penalty_weight = penalty_weight
        self.penalty_scale = penalty_scale
        self.target_head_pr = target_head_pr
        self.target_penalty_mode = target_penalty_mode
        self.penalty_warmup_iters = penalty_warmup_iters
        self.update_rate = update_rate

    def fit(
        self, operator: SlicedModel, values, clean,
        batch_size, factor_batch_size, max_iters, seed=None,
    ):
        if seed is not None:
            torch.manual_seed(seed)
        dimensions = values.shape[-1]
        self.L = F.normalize(torch.randn(dimensions, self.num_factors, device=values.device), dim=0)
        self.R = F.normalize(torch.randn(dimensions, self.num_factors, device=values.device), dim=0)
        self.U = F.normalize(torch.randn(clean.shape[-1], self.num_factors, device=values.device), dim=0)
        self.objective_values = []
        current_penalty_weight = self.penalty_weight

        def statistics(u, left, right, context_values, context_clean):
            target, ov = ordered_cross_hessian_outputs(
                operator, left, right, context_values, context_clean,
            )
            scores = target.float() @ u.float()
            return scores, normalized_head_pr(ov).mean(dim=1)

        def objective(u, left, right, context_values, context_clean):
            scores, ov_pr = statistics(
                u, left, right, context_values, context_clean,
            )
            score_energy = scores.square()
            if self.target_head_pr is None:
                penalty_terms = score_energy * ov_pr
            else:
                pr_penalty = F.relu(ov_pr - self.target_head_pr).square()
                if self.target_penalty_mode == "energy_weighted":
                    penalty_terms = score_energy * pr_penalty
                else:
                    penalty_terms = pr_penalty
            value = (
                0.5 * score_energy.sum()
                - current_penalty_weight * self.penalty_scale * penalty_terms.sum()
            )
            return value, (scores, penalty_terms)

        objective_with_grad = grad_and_value(objective, argnums=(0, 1, 2), has_aux=True)

        def gradients(u, left, right, context_values, context_clean):
            updates, (_, statistics_value) = objective_with_grad(
                u, left, right, context_values, context_clean,
            )
            return tuple(value.detach() for value in statistics_value) + (updates,)

        batched_gradients = vmap(
            gradients, in_dims=(1, 1, 1, None, None),
            out_dims=(1, 1, (1, 1, 1)), chunk_size=factor_batch_size,
        )

        for iteration in trange(max_iters):
            if self.penalty_warmup_iters:
                current_penalty_weight = self.penalty_weight * min(
                    1.0, (iteration + 1) / self.penalty_warmup_iters,
                )
            with torch.no_grad():
                self.L = torch.linalg.qr(self.L).Q
                self.R = torch.linalg.qr(self.R).Q
            score_energy = torch.zeros(self.num_factors, device=values.device)
            penalty_sum = torch.zeros(self.num_factors, device=values.device)
            update_u = torch.zeros_like(self.U)
            update_left = torch.zeros_like(self.L)
            update_right = torch.zeros_like(self.R)
            context_count = 0
            for start in range(0, len(values), batch_size):
                batch_values = values[start:start + batch_size]
                batch_clean = clean[start:start + batch_size]
                scores, penalty_terms, updates = batched_gradients(
                    self.U, self.L, self.R, batch_values, batch_clean,
                )
                with torch.no_grad():
                    score_energy += scores.square().sum(dim=0)
                    penalty_sum += penalty_terms.sum(dim=0)
                    update_u += updates[0]
                    update_left += updates[1]
                    update_right += updates[2]
                context_count += len(batch_values)
            with torch.no_grad():
                target_u = F.normalize(update_u / context_count, dim=0)
                target_left = F.normalize(update_left / context_count, dim=0)
                target_right = F.normalize(update_right / context_count, dim=0)
                self.U = F.normalize(
                    self.U.lerp(target_u, self.update_rate), dim=0,
                )
                self.L = F.normalize(
                    self.L.lerp(target_left, self.update_rate), dim=0,
                )
                self.R = F.normalize(
                    self.R.lerp(target_right, self.update_rate), dim=0,
                )
                mean_energy = score_energy / context_count
                mean_penalty = penalty_sum / context_count
                self.objective_values.append(float(
                    0.5 * mean_energy.sum()
                    - current_penalty_weight * self.penalty_scale * mean_penalty.sum()
                ))
        return self.U, self.L, self.R


def make_operator(model, state, args):
    operator = SlicedModel(
        model, state, args.source_layer, args.target_layer,
        slice(-args.target_positions, None),
    ).to(state[0].device)
    zero = torch.zeros(state[0].shape[-1], device=state[0].device)
    with torch.no_grad():
        clean = operator(zero, zero, state[0], torch.zeros_like(state[0]))[0].detach()
    return clean, operator, AttentionTargetDelta(operator)


def evaluate_attention_pr(dictionary, operator, values, clean, context_batch_size):
    def factor_statistics(left, right, context_values, context_clean):
        _, ov = ordered_cross_hessian_outputs(
            operator, left, right, context_values, context_clean,
        )
        return normalized_head_pr(ov)

    batched = vmap(
        factor_statistics, in_dims=(1, 1, None, None), out_dims=2,
        chunk_size=dictionary.num_factors,
    )
    ov_values = []
    with sdpa_kernel(SDPBackend.MATH):
        for start in range(0, len(values), context_batch_size):
            ov = batched(
                dictionary.L, dictionary.R,
                values[start:start + context_batch_size],
                clean[start:start + context_batch_size],
            )
            ov_values.append(ov)
    ov_values = torch.cat(ov_values)
    return {
        "mean_normalized_ov_pr": float(ov_values.mean()),
        "layer_mean_normalized_ov_pr": ov_values.mean(dim=(0, 2)).detach().cpu().tolist(),
    }


def cross_seed_alignment(reference, candidate):
    similarity = (
        (reference.U.float().T @ candidate.U.float()).abs()
        * (reference.L.float().T @ candidate.L.float()).abs()
        * (reference.R.float().T @ candidate.R.float()).abs()
    ).cpu()
    rows, columns = linear_sum_assignment(similarity.numpy(), maximize=True)
    matched = similarity[rows, columns]
    return {
        "matched_similarity": matched.tolist(),
        "median_similarity": float(matched.median()),
        "maximum_similarity": float(matched.max()),
        "matches_above_0_8": int((matched > 0.8).sum()),
    }


def variants(args):
    yield "baseline", 0.0
    for weight in args.penalty_weights:
        yield f"ov_{weight:g}", weight


def save_dictionary_weights(dictionary, output_directory, seed, variant):
    variant_directory = output_directory / f"seed_{seed}" / variant
    variant_directory.mkdir(parents=True, exist_ok=True)
    for name in ("L", "R", "U"):
        tensor = getattr(dictionary, name).detach().cpu()
        torch.save(tensor, variant_directory / f"{name}.pt")


def run(args):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    train_texts, heldout_texts = read_texts(args.train_contexts, args.heldout_contexts)
    ood_texts = read_ood_texts(args.ood_contexts)
    model, _, metadata = load_tensor_gpt(REPOSITORIES["bilinear"], args.device)
    model.requires_grad_(False)
    states = [
        collect_middle_inputs(model, tokenizer, texts, args.source_layer, args.sequence_length)
        for texts in (train_texts, heldout_texts, ood_texts)
    ]
    operators = [make_operator(model, state, args) for state in states]
    random_asymmetry = random_swap_check(
        operators[0][2], states[0][0][:args.context_batch],
        operators[0][0][:args.context_batch], args.factors, args.fit_seeds[0],
    )
    fits = []
    dictionaries = {}
    variant_specs = list(variants(args))
    for seed in args.fit_seeds:
        seed_results = []
        baseline = AttentionRegularizedBilinearDCT(args.factors, 0.0, 1.0)
        with sdpa_kernel(SDPBackend.MATH):
            baseline.fit(
                operators[0][1], states[0][0], operators[0][0],
                args.fit_context_batch, args.factor_batch, args.iterations, seed,
            )
        penalty_scale = args.penalty_scale if args.penalty_scale is not None else 1.0
        for name, penalty_weight in variant_specs:
            if name == "baseline":
                dictionary = baseline
            else:
                dictionary = AttentionRegularizedBilinearDCT(
                    args.factors, penalty_weight, penalty_scale,
                )
                with sdpa_kernel(SDPBackend.MATH):
                    dictionary.fit(
                        operators[0][1], states[0][0], operators[0][0],
                        args.fit_context_batch, args.factor_batch, args.iterations, seed,
                    )
            save_dictionary_weights(dictionary, args.weights_output, seed, name)
            dictionaries[(seed, name)] = dictionary
            split_results = {}
            for split_name, state, (clean, operator, delta) in zip(
                ("train", "heldout", "ood_wikitext"), states, operators,
            ):
                split_results[split_name] = {
                    **evaluate(dictionary, delta, state[0], clean, args.context_batch),
                    **evaluate_attention_pr(dictionary, operator, state[0], clean, args.context_batch),
                }
            for split_name in ("heldout", "ood_wikitext"):
                split_results[split_name]["energy_retention_vs_train"] = (
                    split_results[split_name]["mean_total_energy"]
                    / split_results["train"]["mean_total_energy"]
                )
            alignment = add_source_unit_diversity(source_weight_alignment(
                dictionary, model.transformer.h[args.source_layer].mlp,
            ))
            seed_results.append({
                "variant": name,
                "penalty_weight": penalty_weight,
                "penalty_scale": penalty_scale,
                "objective_values": dictionary.objective_values,
                **split_results,
                "factor_diversity": factor_diversity(dictionary),
                "source_weight_alignment": alignment,
            })
        fits.append({"fit_seed": seed, "variants": seed_results})
        args.output.write_text(json.dumps({"fits": fits}, indent=2))

    stability = {}
    for name, _ in variant_specs:
        comparisons = []
        for left_index, left_seed in enumerate(args.fit_seeds):
            for right_seed in args.fit_seeds[left_index + 1:]:
                comparisons.append({
                    "left_seed": left_seed,
                    "right_seed": right_seed,
                    **cross_seed_alignment(
                        dictionaries[(left_seed, name)], dictionaries[(right_seed, name)],
                    ),
                })
        stability[name] = comparisons
    result = {
        "description": "Bilinear branch-intervention DCT with energy-weighted attention OV-head PR penalty",
        "architecture": "bilinear",
        "repository": REPOSITORIES["bilinear"],
        "model_metadata": metadata,
        "source_layer": args.source_layer,
        "target_layer": args.target_layer,
        "attention_layers": operators[0][1].attention_layers,
        "fit_seeds": args.fit_seeds,
        "factors": args.factors,
        "iterations": args.iterations,
        "penalty_weights": args.penalty_weights,
        "penalty_mode": "energy_weighted",
        "penalty_scale_override": args.penalty_scale,
        "train_contexts": args.train_contexts,
        "heldout_contexts": args.heldout_contexts,
        "ood_contexts": args.ood_contexts,
        "random_direction_swap_check": random_asymmetry,
        "fits": fits,
        "cross_seed_stability_by_variant": stability,
    }
    args.output.write_text(json.dumps(result, indent=2))
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--source-layer", type=int, default=8)
    parser.add_argument("--target-layer", type=int, default=12)
    parser.add_argument("--fit-seeds", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument("--factors", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--factor-batch", type=int, default=8)
    parser.add_argument("--fit-context-batch", type=int, default=16)
    parser.add_argument("--context-batch", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--target-positions", type=int, default=3)
    parser.add_argument("--train-contexts", type=int, default=32)
    parser.add_argument("--heldout-contexts", type=int, default=64)
    parser.add_argument("--ood-contexts", type=int, default=64)
    parser.add_argument("--penalty-weights", type=float, nargs="+", default=(0.03,))
    parser.add_argument(
        "--penalty-scale", type=float,
        help="Additional energy-weighted PR multiplier (default: 1.0)",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "bilinear_attention_pr_results.json",
    )
    parser.add_argument(
        "--weights-output", type=Path,
        default=ROOT / "bilinear_attention_pr_weights",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())