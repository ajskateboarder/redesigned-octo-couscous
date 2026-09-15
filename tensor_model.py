"""Inference-only modded-nanoGPT model definitions and bilinear algebra."""

from dataclasses import dataclass
import json

import torch
from huggingface_hub import hf_hub_download
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn import functional as F


class CastedLinear(nn.Linear):
    def forward(self, values):
        return F.linear(values, self.weight.to(values.dtype), self.bias)


class Rotary(nn.Module):
    def __init__(self, dimension, base=10000):
        super().__init__()
        inverse_frequency = 1.0 / (base ** (torch.arange(0, dimension, 2).float() / dimension))
        self.register_buffer("inv_freq", inverse_frequency, persistent=False)

    def forward(self, values):
        positions = torch.arange(values.shape[1], device=values.device, dtype=self.inv_freq.dtype)
        frequencies = torch.outer(positions, self.inv_freq)
        return frequencies.cos()[None, :, None], frequencies.sin()[None, :, None]


def apply_rotary(values, cosine, sine):
    half = values.shape[-1] // 2
    left, right = values[..., :half], values[..., half:]
    return torch.cat((left * cosine + right * sine, left * -sine + right * cosine), dim=-1)


@dataclass
class TensorGPTConfig:
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6
    n_embd: int = 768
    squared_mlp: bool = False
    bilinear: bool = False
    expansion_factor: int = 4
    gated: bool = False
    squared_attn: bool = False
    bilinear_attn: bool = False


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.c_q = CastedLinear(config.n_embd, config.n_embd, bias=False)
        self.c_k = CastedLinear(config.n_embd, config.n_embd, bias=False)
        self.c_v = CastedLinear(config.n_embd, config.n_embd, bias=False)
        self.c_proj = CastedLinear(config.n_embd, config.n_embd, bias=False)
        self.rotary = Rotary(self.head_dim)
        self.lamb = nn.Parameter(torch.tensor(.5))

    def forward(self, values, first_values=None):
        batch, sequence, hidden = values.shape
        query = self.c_q(values).view(batch, sequence, self.n_head, self.head_dim)
        key = self.c_k(values).view(batch, sequence, self.n_head, self.head_dim)
        current_values = self.c_v(values).view(batch, sequence, self.n_head, self.head_dim)
        if first_values is None:
            first_values = current_values
        current_values = (1 - self.lamb) * current_values + self.lamb * first_values.view_as(current_values)
        cosine, sine = self.rotary(query)
        query = apply_rotary(F.rms_norm(query, (self.head_dim,)), cosine, sine)
        key = apply_rotary(F.rms_norm(key, (self.head_dim,)), cosine, sine)
        with sdpa_kernel(SDPBackend.MATH):
            output = F.scaled_dot_product_attention(
                query.transpose(1, 2), key.transpose(1, 2), current_values.transpose(1, 2),
                is_causal=True
            )
        return self.c_proj(output.transpose(1, 2).contiguous().view(batch, sequence, hidden)), first_values


class Bilinear(nn.Module):
    def __init__(self, config):
        super().__init__()
        expanded = config.expansion_factor * config.n_embd
        self.Left = CastedLinear(config.n_embd, expanded, bias=False)
        self.Right = CastedLinear(config.n_embd, expanded, bias=False)
        self.Down = CastedLinear(expanded, config.n_embd, bias=False)
        self.Down_bias = nn.Parameter(torch.zeros(config.n_embd))
        self.gated = config.gated

    def forward(self, values):
        left = self.Left(values)
        if self.gated:
            left = F.silu(left)
        return self.Down(left * self.Right(values)) + self.Down_bias

    def pair_effect(self, left_direction, right_direction):
        if self.gated:
            raise ValueError("the exact context-independent pair oracle requires an ungated bilinear module")
        left_right = self.Left(left_direction) * self.Right(right_direction)
        right_left = self.Left(right_direction) * self.Right(left_direction)
        return self.Down(left_right + right_left)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        expanded = 4 * config.n_embd
        self.c_fc = CastedLinear(config.n_embd, expanded, bias=False)
        self.c_proj = CastedLinear(expanded, config.n_embd, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.n_embd))
        self.squared = config.squared_mlp

    def forward(self, values):
        values = self.c_fc(values)
        values = values.square() if self.squared else F.relu(values).square()
        return self.c_proj(values) + self.bias


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.bilinear_attn:
            raise NotImplementedError("bilinear attention checkpoints are not part of the matched MLP comparison")
        self.attn = CausalSelfAttention(config)
        self.mlp = Bilinear(config) if config.bilinear else MLP(config)
        self.lambdas = nn.Parameter(torch.tensor([1., 0.]))

    def forward(self, values, first_values, initial_values):
        values = self.lambdas[0] * values + self.lambdas[1] * initial_values
        attention, first_values = self.attn(F.rms_norm(values, (values.shape[-1],)), first_values)
        values = values + attention
        values = values + self.mlp(F.rms_norm(values, (values.shape[-1],)))
        return values, first_values


class TensorGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        })
        self.lm_head = CastedLinear(config.n_embd, config.vocab_size, bias=False)

    def trace(self, token_ids):
        values = F.rms_norm(self.transformer.wte(token_ids), (self.config.n_embd,))
        states = [values]
        mlp_inputs = []
        initial_values = values
        first_values = None
        for block in self.transformer.h:
            values = block.lambdas[0] * values + block.lambdas[1] * initial_values
            attention, first_values = block.attn(
                F.rms_norm(values, (values.shape[-1],)), first_values
            )
            values = values + attention
            mlp_input = F.rms_norm(values, (values.shape[-1],))
            mlp_inputs.append(mlp_input)
            values = values + block.mlp(mlp_input)
            states.append(values)
        return {"states": states, "mlp_inputs": mlp_inputs}

    def hidden_states(self, token_ids):
        return self.trace(token_ids)["states"]
        return states

    def forward(self, token_ids):
        values = F.rms_norm(self.hidden_states(token_ids)[-1], (self.config.n_embd,))
        return 30 * torch.tanh(self.lm_head(values) / 30)

    def embedding_residual(self, token_ids):
        return F.rms_norm(self.transformer.wte(token_ids), (self.config.n_embd,))

    def state_before_block(self, token_ids, source_layer):
        if not 0 <= source_layer < self.config.n_layer:
            raise ValueError("source_layer must identify an existing block")
        values = self.embedding_residual(token_ids)
        initial_values = values
        first_values = None
        for block in self.transformer.h[:source_layer]:
            values, first_values = block(values, first_values, initial_values)
        return {"values": values, "initial_values": initial_values, "first_values": first_values}


class TensorBlockSpan(nn.Module):
    """Run an exact prefix of complete blocks from the model's embedding residual."""
    def __init__(self, model, depth):
        super().__init__()
        if not 1 <= depth <= model.config.n_layer:
            raise ValueError("depth must select at least one available block")
        self.blocks = model.transformer.h[:depth]

    def forward(self, values):
        initial_values = values
        first_values = None
        for block in self.blocks:
            values, first_values = block(values, first_values, initial_values)
        return values


class TensorMiddleSpan(nn.Module):
    """Continue through internal blocks while holding clean cross-layer state fixed."""
    def __init__(self, model, source_layer, target_layer):
        super().__init__()
        if not 0 < source_layer < target_layer <= model.config.n_layer:
            raise ValueError("middle span must satisfy 0 < source_layer < target_layer <= n_layer")
        self.blocks = model.transformer.h[source_layer:target_layer]

    def forward(self, values, initial_values, first_values):
        for block in self.blocks:
            values, first_values = block(values, first_values, initial_values)
        return values


def load_tensor_gpt(repository, device="cuda"):
    config_path = hf_hub_download(repository, "config.json")
    weights_path = hf_hub_download(repository, "pytorch_model.bin")
    with open(config_path) as handle:
        raw_config = json.load(handle)
    step = raw_config.pop("step", None)
    config = TensorGPTConfig(**raw_config)
    model = TensorGPT(config)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), config, {"step": step, "config_path": config_path, "weights_path": weights_path}


def mixed_difference(function, base, directions):
    result = torch.zeros_like(function(base))
    for subset in range(1 << len(directions)):
        point = base
        width = 0
        for index, direction in enumerate(directions):
            if subset & (1 << index):
                point = point + direction
                width += 1
        result = result + ((-1) ** (len(directions) - width)) * function(point)
    return result