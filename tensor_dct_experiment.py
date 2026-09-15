"""Compare exponential and quadratic DCT interaction transfer on tensor GPT MLPs."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random

import torch
from scipy.stats import spearmanr
from torch import nn
from torch.nn import functional as F
from transformers import GPT2Tokenizer

import dct
from tensor_model import TensorBlockSpan, TensorMiddleSpan, load_tensor_gpt, mixed_difference


ROOT = Path(__file__).resolve().parent
REPOSITORIES = {
    "bilinear": "Elriggs/gpt2-bilinear-18l-9h-1152embd",
    "swiglu": "Elriggs/gpt2-swiglu-18l-9h-1152embd-v2",
}


class LocalMLPDelta(nn.Module):
    def __init__(self, module, radius=1.):
        super().__init__()
        self.module = module
        self.radius = radius
        parameter = next(module.parameters())
        self.device = parameter.device
        self.dtype = parameter.dtype

    def forward(self, theta, inputs, baseline):
        values = self.module(inputs + self.radius * theta) - baseline
        return values[:, -3:].mean(dim=1)


class MiddleSpanOperator(nn.Module):
    def __init__(self, span, initial_values, first_values):
        super().__init__()
        self.span = span
        self.register_buffer("initial_values", initial_values)
        self.register_buffer("first_values", first_values)

    def forward(self, values):
        return self.span(values, self.initial_values, self.first_values)


def read_texts(train_count, heldout_count, transfer_count):
    with (ROOT / "harmful_behaviors.csv").open() as handle:
        behavior = list(dict.fromkeys(row["target"].strip() for row in csv.DictReader(handle)))
    with (ROOT / "harmful_strings.csv").open() as handle:
        transfer = list(dict.fromkeys(row["target"].strip() for row in csv.DictReader(handle)))
    random.Random(1729).shuffle(behavior)
    random.Random(2718).shuffle(transfer)
    return (behavior[:train_count], behavior[train_count:train_count + heldout_count],
            transfer[:transfer_count])


def collect_mlp_inputs(model, tokenizer, texts, layer, sequence_length, batch_size):
    inputs, hashes = [], []
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(texts[start:start + batch_size], return_tensors="pt", truncation=True,
                            padding="max_length", max_length=sequence_length).input_ids.to(next(model.parameters()).device)
        hashes.extend(hashlib.sha256(str(row.tolist()).encode()).hexdigest() for row in encoded.cpu())
        with torch.no_grad():
            inputs.append(model.trace(encoded)["mlp_inputs"][layer].float())
    return torch.cat(inputs), hashes


def collect_embedding_inputs(model, tokenizer, texts, sequence_length, batch_size):
    inputs, hashes = [], []
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(texts[start:start + batch_size], return_tensors="pt", truncation=True,
                            padding="max_length", max_length=sequence_length).input_ids.to(next(model.parameters()).device)
        hashes.extend(hashlib.sha256(str(row.tolist()).encode()).hexdigest() for row in encoded.cpu())
        with torch.no_grad():
            inputs.append(model.embedding_residual(encoded).float())
    return torch.cat(inputs), hashes


def collect_middle_inputs(model, tokenizer, texts, source_layer, sequence_length):
    values, initial_values, first_values, hashes = [], [], [], []
    for text in texts:
        encoded = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length",
                            max_length=sequence_length).input_ids.to(next(model.parameters()).device)
        hashes.append(hashlib.sha256(str(encoded[0].cpu().tolist()).encode()).hexdigest())
        with torch.no_grad():
            state = model.state_before_block(encoded, source_layer)
        values.append(state["values"].float())
        initial_values.append(state["initial_values"].float())
        first_values.append(state["first_values"].float())
    return torch.cat(values), torch.cat(initial_values), torch.cat(first_values), hashes


def exponential_pair_vectors(U, V, directions, coordinate_scale):
    slopes = coordinate_scale * (V.T @ directions)
    pairs = torch.triu_indices(directions.shape[1], directions.shape[1], offset=1)
    coefficients = torch.expm1(slopes[:, pairs[0]]) * torch.expm1(slopes[:, pairs[1]])
    return U @ coefficients, pairs


def quadratic_pair_vectors(U, V, directions, coordinate_scale):
    slopes = coordinate_scale * (V.T @ directions)
    pairs = torch.triu_indices(directions.shape[1], directions.shape[1], offset=1)
    coefficients = slopes[:, pairs[0]] * slopes[:, pairs[1]]
    return U @ coefficients, pairs


def calibrated_quadratic_pair_vectors(U, V, amplitudes, directions, coordinate_scale):
    return quadratic_pair_vectors(U * amplitudes.unsqueeze(0), V, directions, coordinate_scale)


def contextual_mode_effects(module, inputs, left, right, radius, coordinate_scale):
    baseline = module(inputs)
    values = []
    for factor in range(left.shape[1]):
        left_delta = radius * coordinate_scale * left[:, factor]
        right_delta = radius * coordinate_scale * right[:, factor]
        left_output = module(inputs + left_delta) - baseline
        right_output = module(inputs + right_delta) - baseline
        joint = module(inputs + left_delta + right_delta) - baseline
        values.append((joint - left_output - right_output)[:, -3:].mean(dim=1))
    return torch.stack(values, dim=2)


def random_contextual_modes(module, train_inputs, factors, radius, coordinate_scale,
                            seed, device):
    generator = torch.Generator(device=device).manual_seed(seed)
    hidden = train_inputs.shape[-1]
    left, _ = torch.linalg.qr(torch.randn(hidden, factors, generator=generator, device=device))
    right, _ = torch.linalg.qr(torch.randn(hidden, factors, generator=generator, device=device))
    with torch.no_grad():
        interactions = contextual_mode_effects(
            module, train_inputs, left, right, radius, coordinate_scale
        )
    output_directions = []
    context_scores = []
    for factor in range(factors):
        _, _, right_vectors = torch.linalg.svd(interactions[:, :, factor].float(), full_matrices=False)
        output = right_vectors[0]
        scores = interactions[:, :, factor].float() @ output
        output_directions.append(output)
        context_scores.append(scores)
    context_scores = torch.stack(context_scores, dim=1)
    return {
        "U": torch.stack(output_directions, dim=1),
        "L": left,
        "R": right,
        "context_scores": context_scores,
        "amplitudes": context_scores.square().mean(dim=0).sqrt(),
        "signed_amplitudes": context_scores.mean(dim=0),
    }


def rank_correlation(left, right):
    left = torch.as_tensor(left).flatten().cpu()
    right = torch.as_tensor(right).flatten().cpu()
    if len(left) < 2 or left.std() == 0 or right.std() == 0:
        return 0.
    value = float(spearmanr(left.numpy(), right.numpy()).statistic)
    return value if torch.isfinite(torch.tensor(value)) else 0.


def contextual_metrics(output_directions, train_amplitudes, train_signed_amplitudes,
                       coordinate_scale, actual):
    directions = F.normalize(output_directions.float(), dim=0).cpu()
    actual = actual.float().cpu()
    actual_norm = actual.norm(dim=1)
    cosine = torch.einsum("cof,of->cf", F.normalize(actual, dim=1), directions)
    projection = torch.einsum("cof,of->cf", actual, directions)
    energy_fraction = projection.square() / actual_norm.square().clamp_min(1e-12)
    train_amplitudes = train_amplitudes.float().cpu()
    actual_rms = projection.square().mean(dim=0).sqrt()
    predicted = directions * (coordinate_scale ** 2 * train_signed_amplitudes.float().cpu()).unsqueeze(0)
    predicted = predicted.unsqueeze(0).expand(len(actual), -1, -1)
    relative_error = (predicted - actual).norm(dim=1) / actual_norm.clamp_min(1e-12)
    norm_ratio = predicted.norm(dim=1) / actual_norm.clamp_min(1e-12)
    return {"absolute_cosine": quantiles(cosine.abs()),
            "signed_cosine": quantiles(cosine),
            "projected_energy_fraction": quantiles(energy_fraction),
            "amplitude_spearman": rank_correlation(train_amplitudes, actual_rms),
            "sign_consistency": quantiles(projection.mean(dim=0).abs() /
                                           projection.square().mean(dim=0).sqrt().clamp_min(1e-12)),
            "calibrated_relative_error": quantiles(relative_error),
            "calibrated_norm_ratio": quantiles(norm_ratio),
            "actual_norm": quantiles(actual_norm)}


def actual_pair_vectors(module, inputs, directions, radius, coordinate_scale):
    baseline = module(inputs)
    singleton = []
    for index in range(directions.shape[1]):
        singleton.append(module(inputs + radius * coordinate_scale * directions[:, index]) - baseline)
    values = []
    pairs = torch.triu_indices(directions.shape[1], directions.shape[1], offset=1)
    for left, right in pairs.T.tolist():
        joint = module(inputs + radius * coordinate_scale * (directions[:, left] + directions[:, right])) - baseline
        interaction = joint - singleton[left] - singleton[right]
        values.append(interaction[:, -3:].mean(dim=1).mean(dim=0))
    return torch.stack(values, dim=1), pairs


def bilinear_oracle_vectors(module, directions, radius, coordinate_scale):
    pairs = torch.triu_indices(directions.shape[1], directions.shape[1], offset=1)
    values = [module.pair_effect(radius * coordinate_scale * directions[:, left],
                                 radius * coordinate_scale * directions[:, right])
              for left, right in pairs.T.tolist()]
    return torch.stack(values, dim=1), pairs


def vector_metrics(predicted, actual):
    predicted = predicted.float().cpu()
    actual = actual.float().cpu()
    cosine = F.cosine_similarity(predicted.T, actual.T, dim=1)
    predicted_norm = predicted.norm(dim=0)
    actual_norm = actual.norm(dim=0)
    gain_rho = rank_correlation(predicted_norm, actual_norm)
    return {"cosine": quantiles(cosine), "gain_spearman": gain_rho,
            "nonzero_fraction": float((actual_norm > 1e-8).float().mean())}


def target_metrics(predicted, actual, targets, top_k=5):
    targets = F.normalize(targets.float(), dim=0).cpu()
    predicted_scores = targets.T @ predicted.float().cpu()
    actual_scores = targets.T @ actual.float().cpu()
    rank = rank_correlation(predicted_scores, actual_scores)
    selected_actual = []
    for target in range(targets.shape[1]):
        selected = torch.topk(predicted_scores[target], min(top_k, predicted_scores.shape[1])).indices
        selected_actual.extend(actual_scores[target, selected].tolist())
    selected_actual = torch.tensor(selected_actual)
    return {"spearman": rank, "top_k": min(top_k, predicted_scores.shape[1]),
            "top_actual": quantiles(selected_actual),
            "all_actual_mean": float(actual_scores.mean())}


def quantiles(values):
    values = values.float().flatten()
    return {name: float(torch.quantile(values, probability)) for name, probability in
            (("min", 0.), ("p10", .1), ("median", .5), ("p90", .9), ("max", 1.))}


def save_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def run(args):
    torch.manual_seed(args.seed)
    repository = REPOSITORIES[args.architecture]
    model, config, metadata = load_tensor_gpt(repository, args.device)
    model.requires_grad_(False)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    train_texts, heldout_texts, transfer_texts = read_texts(
        args.train_contexts, args.heldout_contexts, args.transfer_contexts)
    if args.operator == "mlp":
        collect = lambda texts: collect_mlp_inputs(
            model, tokenizer, texts, args.layer, args.sequence_length, args.context_batch
        )
        module = model.transformer.h[args.layer].mlp
    elif args.operator == "span":
        collect = lambda texts: collect_embedding_inputs(
            model, tokenizer, texts, args.sequence_length, args.context_batch
        )
        module = TensorBlockSpan(model, args.depth)
    else:
        train, train_initial, train_first, train_hashes = collect_middle_inputs(
            model, tokenizer, train_texts, args.source_layer, args.sequence_length
        )
        heldout, heldout_initial, heldout_first, heldout_hashes = collect_middle_inputs(
            model, tokenizer, heldout_texts, args.source_layer, args.sequence_length
        )
        transfer, transfer_initial, transfer_first, transfer_hashes = collect_middle_inputs(
            model, tokenizer, transfer_texts, args.source_layer, args.sequence_length
        )
        span = TensorMiddleSpan(model, args.source_layer, args.target_layer)
        modules = {
            "train": MiddleSpanOperator(span, train_initial, train_first),
            "heldout": MiddleSpanOperator(span, heldout_initial, heldout_first),
            "transfer": MiddleSpanOperator(span, transfer_initial, transfer_first),
        }
        module = modules["train"]
    if args.operator != "middle":
        train, train_hashes = collect(train_texts)
        heldout, heldout_hashes = collect(heldout_texts)
        transfer, transfer_hashes = collect(transfer_texts)
    if len(set(train_hashes + heldout_hashes + transfer_hashes)) != len(train_hashes + heldout_hashes + transfer_hashes):
        raise ValueError("tokenized context splits overlap")
    with torch.no_grad():
        train_baseline = module(train)
    fit_batch_size = len(train) if args.operator == "middle" else 1
    if args.radius is None:
        physical_delta = LocalMLPDelta(module)
        torch.manual_seed(args.calibration_seed)
        radius = dct.SteeringCalibrator(target_ratio=.5).calibrate(
            physical_delta, train, train_baseline, batch_size=fit_batch_size,
            calibration_sample_size=args.calibration_directions, factor_batch_size=args.factor_batch)
    else:
        radius = args.radius
    coordinate_delta = LocalMLPDelta(module, radius)

    torch.manual_seed(args.seed)
    exponential = dct.ExponentialDCT(num_factors=args.factors)
    exp_U, exp_V = exponential.fit(coordinate_delta, train, train_baseline, batch_size=fit_batch_size,
                                   factor_batch_size=args.factor_batch, input_scale=1.,
                                   max_iters=args.iterations, init="random")
    torch.manual_seed(args.seed)
    quadratic = dct.QuadraticDCT(num_factors=args.factors)
    quad_U, quad_V = quadratic.fit(coordinate_delta, train, train_baseline, batch_size=fit_batch_size,
                                   factor_batch_size=args.factor_batch, max_iters=args.iterations,
                                   init="random")
    contextual = None
    contextual_null = None
    if args.contextual:
        torch.manual_seed(args.seed)
        contextual = dct.ContextualQuadraticDCT(num_factors=args.factors)
        context_U, context_L, context_R = contextual.fit(
            coordinate_delta, train, train_baseline, batch_size=fit_batch_size,
            factor_batch_size=args.factor_batch, max_iters=args.iterations,
        )
        contextual_null = random_contextual_modes(
            module, train, args.factors, radius, args.coordinate_scale,
            args.null_seed, args.device,
        )
    directions = F.normalize(exp_V.detach(), dim=0)
    exp_prediction, pairs = exponential_pair_vectors(exp_U.detach(), exp_V.detach(), directions,
                                                     args.coordinate_scale)
    quad_prediction, quad_pairs = quadratic_pair_vectors(quad_U.detach(), quad_V.detach(), directions,
                                                         args.coordinate_scale)
    calibrated_quad_prediction, calibrated_pairs = calibrated_quadratic_pair_vectors(
        quad_U.detach(), quad_V.detach(), quadratic.amplitudes.detach(), directions,
        args.coordinate_scale,
    )
    if not torch.equal(pairs, quad_pairs) or not torch.equal(pairs, calibrated_pairs):
        raise RuntimeError("predictors generated different pair orderings")
    results = {}
    raw = {"pairs": pairs.cpu(), "directions": directions.cpu(),
            "exp_prediction": exp_prediction.cpu(), "quad_prediction": quad_prediction.cpu(),
            "calibrated_quad_prediction": calibrated_quad_prediction.cpu(),
            "quad_amplitudes": quadratic.amplitudes.detach().cpu()}
    for domain, inputs in (("heldout", heldout), ("transfer", transfer)):
        evaluation_module = modules[domain] if args.operator == "middle" else module
        with torch.no_grad():
            actual, actual_pairs = actual_pair_vectors(evaluation_module, inputs, directions, radius,
                                                       args.coordinate_scale)
        if not torch.equal(pairs, actual_pairs):
            raise RuntimeError("actual pair order does not match predictions")
        domain_result = {"exponential": {**vector_metrics(exp_prediction, actual),
                          "target": target_metrics(exp_prediction, actual, exp_U.detach())},
                 "quadratic": {**vector_metrics(quad_prediction, actual),
                        "target": target_metrics(quad_prediction, actual, quad_U.detach())},
                         "quadratic_calibrated": {
                             **vector_metrics(calibrated_quad_prediction, actual),
                             "target": target_metrics(calibrated_quad_prediction, actual, quad_U.detach()),
                         },
                         "actual_norm": quantiles(actual.norm(dim=0))}
        if contextual is not None:
            context_actual = contextual_mode_effects(
                evaluation_module, inputs, context_L.detach(), context_R.detach(), radius,
                args.coordinate_scale,
            )
            domain_result["contextual_quadratic"] = contextual_metrics(
                context_U.detach(), contextual.amplitudes.detach(),
                contextual.signed_amplitudes.detach(), args.coordinate_scale, context_actual,
            )
            raw[f"contextual_actual_{domain}"] = context_actual.cpu()
            null_actual = contextual_mode_effects(
                evaluation_module, inputs, contextual_null["L"], contextual_null["R"],
                radius, args.coordinate_scale,
            )
            domain_result["contextual_random_null"] = contextual_metrics(
                contextual_null["U"], contextual_null["amplitudes"],
                contextual_null["signed_amplitudes"], args.coordinate_scale, null_actual,
            )
            raw[f"contextual_null_actual_{domain}"] = null_actual.cpu()
        if args.architecture == "bilinear" and args.operator == "mlp":
            oracle, oracle_pairs = bilinear_oracle_vectors(module, directions, radius,
                                                           args.coordinate_scale)
            if not torch.equal(pairs, oracle_pairs):
                raise RuntimeError("oracle pair order does not match predictions")
            domain_result["oracle"] = vector_metrics(oracle, actual)
            raw["oracle"] = oracle.cpu()
        results[domain] = domain_result
        raw[f"actual_{domain}"] = actual.cpu()
    if args.operator == "mlp":
        operator_name = f"mlp_{args.layer}"
    elif args.operator == "span":
        operator_name = f"span_{args.depth}"
    else:
        operator_name = f"middle_{args.source_layer}_{args.target_layer}"
    output = args.output / args.architecture / operator_name
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"exp_U": exp_U.detach().cpu(), "exp_V": exp_V.detach().cpu(),
                "quad_U": quad_U.detach().cpu(), "quad_V": quad_V.detach().cpu(),
                **({"context_U": context_U.detach().cpu(), "context_L": context_L.detach().cpu(),
                    "context_R": context_R.detach().cpu(),
                    "context_amplitudes": contextual.amplitudes.detach().cpu(),
                    "context_signed_amplitudes": contextual.signed_amplitudes.detach().cpu()}
                   if contextual is not None else {}),
                **({"null_U": contextual_null["U"].cpu(),
                    "null_L": contextual_null["L"].cpu(),
                    "null_R": contextual_null["R"].cpu(),
                    "null_amplitudes": contextual_null["amplitudes"].cpu(),
                    "null_signed_amplitudes": contextual_null["signed_amplitudes"].cpu()}
                   if contextual_null is not None else {}),
                "radius": radius, **raw}, output / "raw.pt")
    report = {"protocol": {"repository": repository, "architecture": args.architecture,
                            "config": vars(config), "step": metadata["step"],
                            "operator": args.operator, "layer": args.layer, "depth": args.depth,
                            "source_layer": args.source_layer, "target_layer": args.target_layer,
                            "calibration_seed": args.calibration_seed,
                            "null_seed": args.null_seed,
                            "factors": args.factors, "iterations": args.iterations,
                            "coordinate_scale": args.coordinate_scale, "radius": radius,
                            "radius_source": "override" if args.radius is not None else "calibrated",
                            "train_hashes": train_hashes, "heldout_hashes": heldout_hashes,
                            "transfer_hashes": transfer_hashes}, "results": results}
    save_json(output / "report.json", report)
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=REPOSITORIES, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=ROOT / "tensor_dct_results" / "v1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--calibration-seed", type=int, default=0)
    parser.add_argument("--null-seed", type=int, default=10000)
    parser.add_argument("--operator", choices=("mlp", "span", "middle"), default="mlp")
    parser.add_argument("--layer", type=int, default=9)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--source-layer", type=int, default=5)
    parser.add_argument("--target-layer", type=int, default=13)
    parser.add_argument("--factors", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--factor-batch", type=int, default=16)
    parser.add_argument("--calibration-directions", type=int, default=12)
    parser.add_argument("--radius", type=float)
    parser.add_argument("--contextual", action="store_true")
    parser.add_argument("--coordinate-scale", type=float, default=.5)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--context-batch", type=int, default=2)
    parser.add_argument("--train-contexts", type=int, default=8)
    parser.add_argument("--heldout-contexts", type=int, default=16)
    parser.add_argument("--transfer-contexts", type=int, default=16)
    run(parser.parse_args())


if __name__ == "__main__":
    main()