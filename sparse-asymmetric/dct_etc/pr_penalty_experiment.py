import argparse
import json
from pathlib import Path

import torch
from torch import vmap
from torch.func import grad
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import GPT2Tokenizer

from dct import AsymmetricQuadraticDCT, directional_hessian_output
from intermediate_mlp_pr_experiment import (
    IntermediateMLPFeatures,
    MiddleSpanOperator,
    REPOSITORIES,
    SpanDelta,
    collect_middle_inputs,
    participation_ratio,
    read_texts,
)
from ..tensor_model import load_tensor_gpt


ROOT = Path(__file__).resolve().parent


class ParticipationRegularizedDCT:
    def __init__(self, num_factors, penalty_weight, penalty_scale):
        self.num_factors = num_factors
        self.penalty_weight = penalty_weight
        self.penalty_scale = penalty_scale

    def fit(
        self, delta, features, X, Y, batch_size=1, factor_batch_size=16,
        max_iters=5, beta=1.0,
    ):
        self.num_samples, _, self.d_source = X.shape
        self.device = delta.device
        self.L = F.normalize(torch.randn(
            self.d_source, self.num_factors, device=self.device, dtype=torch.float32,
        ), dim=0)
        self.R = F.normalize(torch.randn(
            self.d_source, self.num_factors, device=self.device, dtype=torch.float32,
        ), dim=0)
        self.U = F.normalize(torch.randn(
            Y.shape[-1], self.num_factors, device=self.device, dtype=torch.float32,
        ), dim=0)
        feature_dimensions = features(
            torch.zeros(self.d_source, device=self.device), X[:1], Y[:1],
        ).shape[-1]
        self.objective_values = []
        self.score_energy_values = []
        self.normalized_pr_values = []

        def statistics(u, left, right, context_values, context_clean):
            target_response = directional_hessian_output(
                delta, left, right, context_values, context_clean,
            )
            intermediate_response = directional_hessian_output(
                features, left, right, context_values, context_clean,
            )
            score = u.float() @ target_response.float()
            normalized_pr = participation_ratio(intermediate_response) / feature_dimensions
            return score, normalized_pr

        def gradients(u, left, right, context_values, context_clean):
            def objective(current_u, current_left, current_right):
                score, normalized_pr = statistics(
                    current_u, current_left, current_right,
                    context_values, context_clean,
                )
                penalty = self.penalty_weight * self.penalty_scale * normalized_pr
                return 0.5 * score.square() - penalty

            score, normalized_pr = statistics(
                u, left, right, context_values, context_clean,
            )
            updates = grad(objective, argnums=(0, 1, 2))(u, left, right)
            return score.detach(), normalized_pr.detach(), updates

        batched_gradients = vmap(
            gradients, in_dims=(1, 1, 1, None, None),
            out_dims=(0, 0, (1, 1, 1)), chunk_size=factor_batch_size,
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
                for context in range(len(context_values)):
                    scores, normalized_pr, updates = batched_gradients(
                        self.U, self.L, self.R,
                        context_values[context:context + 1],
                        context_clean[context:context + 1],
                    )
                    with torch.no_grad():
                        score_energy += scores.square()
                        normalized_pr_sum += normalized_pr
                        update_u += updates[0]
                        update_left += updates[1]
                        update_right += updates[2]
                    context_count += 1
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
    features = IntermediateMLPFeatures(
        model, state, args.source_layer, args.target_layer,
        slice(-args.target_positions, None),
    )
    return clean, delta, features


def evaluate(dictionary, delta, features, values, clean):
    energies = []
    participation_ratios = []
    with sdpa_kernel(SDPBackend.MATH):
        for context in range(len(values)):
            context_energies = []
            context_pr = []
            for factor in range(dictionary.num_factors):
                target_response = directional_hessian_output(
                    delta, dictionary.L[:, factor], dictionary.R[:, factor],
                    values[context:context + 1], clean[context:context + 1],
                )
                score = dictionary.U[:, factor].float() @ target_response.float()
                intermediate_response = directional_hessian_output(
                    features, dictionary.L[:, factor], dictionary.R[:, factor],
                    values[context:context + 1], clean[context:context + 1],
                )
                context_energies.append(score.square())
                context_pr.append(participation_ratio(intermediate_response))
            energies.append(torch.stack(context_energies))
            participation_ratios.append(torch.stack(context_pr))
    energies = torch.stack(energies)
    participation_ratios = torch.stack(participation_ratios)
    return {
        "factor_context_energy": energies.detach().cpu().tolist(),
        "mean_total_energy": float(energies.sum(dim=1).mean()),
        "median_participation_ratio": float(participation_ratios.median()),
        "mean_participation_ratio": float(participation_ratios.mean()),
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
    for seed in args.fit_seeds:
        torch.manual_seed(seed)
        baseline = AsymmetricQuadraticDCT(num_factors=args.factors)
        with sdpa_kernel(SDPBackend.MATH):
            baseline.fit(
                train_delta, states[0][0], train_clean, batch_size=1,
                factor_batch_size=args.factor_batch, max_iters=args.iterations,
            )
        baseline_train = evaluate(
            baseline, train_delta, train_features, states[0][0], train_clean,
        )
        penalty_scale = baseline_train["mean_total_energy"] / args.factors
        sweep = []
        for penalty_weight in args.penalty_weights:
            if penalty_weight == 0:
                dictionary = baseline
            else:
                torch.manual_seed(seed)
                dictionary = ParticipationRegularizedDCT(
                    args.factors, penalty_weight, penalty_scale,
                )
                with sdpa_kernel(SDPBackend.MATH):
                    dictionary.fit(
                        train_delta, train_features, states[0][0], train_clean,
                        batch_size=1, factor_batch_size=args.factor_batch,
                        max_iters=args.iterations,
                    )
            sweep.append({
                "penalty_weight": penalty_weight,
                "penalty_scale": penalty_scale,
                "train": baseline_train if penalty_weight == 0 else evaluate(
                    dictionary, train_delta, train_features, states[0][0], train_clean,
                ),
                "heldout": evaluate(
                    dictionary, heldout_delta, heldout_features,
                    states[1][0], heldout_clean,
                ),
            })
        fits.append({"fit_seed": seed, "sweep": sweep})
    return {"repository": REPOSITORIES[architecture], "model_metadata": metadata, "fits": fits}


def run(args):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    texts = read_texts(args.train_contexts, args.heldout_contexts)
    result = {
        "description": "Direct intermediate-MLP PR-penalty sweep",
        "source_layer": args.source_layer,
        "target_layer": args.target_layer,
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
    parser.add_argument("--factors", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--factor-batch", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=32)

if __name__ == "__main__":
    run(parse_args())