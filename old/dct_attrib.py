import gc
import math
from typing import Optional

from jaxtyping import Float, Int
from tqdm.auto import tqdm
from torch import vmap, Tensor
import torch

def _streamed_combinations(x, r, batch_size=100000):
    n, dt, dev = len(x), x.dtype, x.device
    if r == 1:
        for i in range(0, n, batch_size): yield x[i:i+batch_size].unsqueeze(-1)
        return
    if r == 2:
        p = torch.triu_indices(n, n, 1, dtype=torch.long, device=dev).t()
        for i in range(0, len(p), batch_size): yield x[p[i:i+batch_size].long()]
        return
    
    step = max(1, 5000000 // (n ** (r - 1)))
    buf = []

    for s in range(0, n - r + 1, step):
        grid = torch.meshgrid([torch.arange(s, min(s + step, n - r + 1), dtype=dt, device=dev)] + [torch.arange(n, dtype=dt, device=dev)] * (r - 1), indexing='ij')
        gf = torch.stack(grid, dim=-1).reshape(-1, r)
        idxs = gf[torch.all(gf[:, :-1] < gf[:, 1:], dim=-1)]

        if len(idxs) > 0: buf.append(x[idxs.long()])

        while sum(len(c) for c in buf) >= batch_size:
            res = torch.cat(buf, dim=0)
            yield res[:batch_size].detach()
            buf = [res[batch_size:]] if len(res) > batch_size else [] 
 
    if buf:
        res = torch.cat(buf, dim=0)
        if len(res) > 0: yield res

class StreamedTopK:
    def __init__(self, top_k=10, mode="top", device="cpu"):
        self.top_k = top_k
        self.mode = mode
        self.device = device

        self.top_values = None
        self.top_combos = None

    def update(self, combos: torch.Tensor, values: torch.Tensor):
        score_values = values if self.mode == "top" else -values

        if self.top_values is None:
            k = min(self.top_k, values.shape[0])
            _, best_idx = torch.topk(score_values, k=k, largest=True)
            self.top_values = values[best_idx]
            self.top_combos = combos[best_idx]
        else:
            combined_vals = torch.cat([self.top_values if self.mode == "top" else -self.top_values, score_values])
            combined_combos = torch.cat([self.top_combos, combos], dim=0)

            k = min(self.top_k, combined_vals.shape[0])
            _, best_idx = torch.topk(combined_vals, k=k, largest=True)
            
            self.top_values = combined_vals[best_idx] if self.mode == "top" else -combined_vals[best_idx]
            self.top_combos = combined_combos[best_idx]

    def get(self):
        # Ensure output is sorted
        sorted_idx = torch.argsort(self.top_values, descending=(self.mode == "top"))
        return self.top_combos[sorted_idx], self.top_values[sorted_idx]

class DCTAttrib:
    def __init__(self, V: Float[Tensor, "features batch"], U: Float[Tensor, "features batch"], device: torch.device):
        self.V = V
        self.U = U
        self.V_dot = V.T @ V
        self.U_dot = U.T @ U
        self.device = device
        self.d = V.shape[0]

    def _I(self, j: int, S: Int[Tensor, "indices"], c: int = 1):
        p = torch.arange(0, self.V.shape[1])
        expm = torch.expm1(c*self.V_dot)
        u_dot = self.U_dot[:, j]

        def _I_single_edge(l):
            l = l.long()
            ul_uj = u_dot[l]
            dp = vmap(lambda i: expm[l, i.long()])(S)
            k = torch.prod(dp)
            return ul_uj * k

        return torch.sum(vmap(_I_single_edge)(p))

    def _many_to_one(self, j: int, S: Int[Tensor, "indices"], c: float = 1):
        responses = self.U @ torch.expm1(c * self.V_dot)
        target = responses[:, j]
        target = target / target.norm().clamp_min(1e-12)
        source_responses = responses[:, S.long()]
        source_responses = source_responses / source_responses.norm(dim=0, keepdim=True).clamp_min(1e-12)
        return (target @ source_responses).sum()

    def _interaction_to_one(self, j: int, S: Int[Tensor, "indices"], c: float = 1):
        expm = torch.expm1(c * self.V_dot)
        target = self.U @ expm[:, j]
        interaction = self.U @ torch.prod(expm[:, S.long()], dim=1)
        return (target @ interaction) / (target.norm() * interaction.norm()).clamp_min(1e-12)

    def I(self, j: int, width: int, batch_size: int, k: int, input_scale: int = 1, silent=False):
        device = self.device

        try:
            top_k, bottom_k = StreamedTopK(top_k=k), StreamedTopK(top_k=k, mode="bottom")
            x = torch.arange(self.V.shape[1], device=device, dtype=torch.int16)
            S = _streamed_combinations(x, width, batch_size)
            S_len = math.comb(self.V.shape[1], width)

            with tqdm(total=S_len, disable=silent) as pbar:
                for batch in S:
                    batch = batch.to(device)
                    g = vmap(self._I, in_dims=(None, 0, None))(j, batch, input_scale).detach().cpu()
                    top_k.update(batch, g)
                    bottom_k.update(batch, g)
                    pbar.update(batch.shape[0])
        finally:
            gc.collect()
            torch.cuda.empty_cache()

        return top_k, bottom_k

    def many_to_one(self, j: int, width: int, batch_size: int, k: int,
                    input_scale: float = 1, silent=False):
        """Find source-factor sets whose singleton responses converge on factor j's response mode."""
        if not 0 <= j < self.V.shape[1]:
            raise ValueError("j is outside the factor dictionary")
        candidates = torch.arange(self.V.shape[1], device=self.device, dtype=torch.int64)
        candidates = candidates[candidates != j]
        if not 1 <= width <= len(candidates):
            raise ValueError("width must select at least one non-target factor")

        responses = self.U @ torch.expm1(input_scale * self.V_dot)
        responses = responses / responses.norm(dim=0, keepdim=True).clamp_min(1e-12)
        affinities = responses[:, j] @ responses

        try:
            top_k = StreamedTopK(top_k=k)
            bottom_k = StreamedTopK(top_k=k, mode="bottom")
            combinations = _streamed_combinations(candidates, width, batch_size)
            total = math.comb(len(candidates), width)
            with tqdm(total=total, disable=silent) as pbar:
                for batch in combinations:
                    scores = affinities[batch.long()].sum(dim=1).detach().cpu()
                    batch = batch.cpu()
                    top_k.update(batch, scores)
                    bottom_k.update(batch, scores)
                    pbar.update(len(batch))
        finally:
            gc.collect()
            torch.cuda.empty_cache()

        return top_k, bottom_k

    def aligned_connection(self, j: int, k: int, alignment_weight: float = .2,
                           input_scale: float = 1):
        """Rank singleton edges by strength with a bounded response-alignment bonus."""
        if not 0 <= j < self.V.shape[1]:
            raise ValueError("j is outside the factor dictionary")
        if alignment_weight < 0:
            raise ValueError("alignment_weight must be non-negative")

        candidates = torch.arange(self.V.shape[1], device=self.device, dtype=torch.int64)
        candidates = candidates[candidates != j]
        responses = self.U @ torch.expm1(input_scale * self.V_dot)
        strengths = self.U[:, j] @ responses[:, candidates]
        unit_responses = responses / responses.norm(dim=0, keepdim=True).clamp_min(1e-12)
        alignments = unit_responses[:, j] @ unit_responses[:, candidates]

        strength_ranks = torch.argsort(torch.argsort(strengths, stable=True), stable=True)
        alignment_ranks = torch.argsort(torch.argsort(alignments, stable=True), stable=True)
        denominator = max(1, len(candidates) - 1)
        scores = (strength_ranks + alignment_weight * alignment_ranks) / denominator

        top_k = StreamedTopK(top_k=k)
        bottom_k = StreamedTopK(top_k=k, mode="bottom")
        singleton_candidates = candidates.unsqueeze(1).cpu()
        scores = scores.detach().cpu()
        top_k.update(singleton_candidates, scores)
        bottom_k.update(singleton_candidates, scores)
        return top_k, bottom_k

    def interaction_to_one(self, j: int, width: int, batch_size: int, k: int,
                           input_scale: float = 1, silent=False):
        """Rank frozen-surrogate source interactions by alignment with factor j's response mode."""
        if not 0 <= j < self.V.shape[1]:
            raise ValueError("j is outside the factor dictionary")
        candidates = torch.arange(self.V.shape[1], device=self.device, dtype=torch.int64)
        candidates = candidates[candidates != j]
        if not 2 <= width <= len(candidates):
            raise ValueError("width must select at least two non-target factors")

        expm = torch.expm1(input_scale * self.V_dot)
        target = self.U @ expm[:, j]
        target = target / target.norm().clamp_min(1e-12)

        try:
            top_k = StreamedTopK(top_k=k)
            bottom_k = StreamedTopK(top_k=k, mode="bottom")
            combinations = _streamed_combinations(candidates, width, batch_size)
            total = math.comb(len(candidates), width)
            with tqdm(total=total, disable=silent) as pbar:
                for batch in combinations:
                    atom_coefficients = torch.prod(expm[:, batch.long()], dim=2)
                    interactions = self.U @ atom_coefficients
                    interactions /= interactions.norm(dim=0, keepdim=True).clamp_min(1e-12)
                    scores = (target @ interactions).detach().cpu()
                    batch = batch.cpu()
                    top_k.update(batch, scores)
                    bottom_k.update(batch, scores)
                    pbar.update(len(batch))
        finally:
            gc.collect()
            torch.cuda.empty_cache()

        return top_k, bottom_k