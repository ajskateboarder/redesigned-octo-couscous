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

    def I_vectors(self, j: int, vector_sets: Float[Tensor, "... width features"], input_scale: float = 1):
        if vector_sets.ndim not in (2, 3):
            raise ValueError("vector_sets must have shape (width, features) or (batch, width, features)")
        if vector_sets.shape[-1] != self.d:
            raise ValueError(
                f"vector dimension {vector_sets.shape[-1]} does not match V dimension {self.d}"
            )

        squeeze_result = vector_sets.ndim == 2
        if squeeze_result:
            vector_sets = vector_sets.unsqueeze(0)

        vector_sets = vector_sets.to(device=self.V.device, dtype=self.V.dtype)
        edge_terms = torch.expm1(input_scale * (vector_sets @ self.V)).prod(dim=1)
        scores = edge_terms @ self.U_dot[:, j]
        return scores.squeeze(0) if squeeze_result else scores

    def permutation_z_score(
        self,
        j: int,
        S: Float[Tensor, "width features"],
        null_vectors: Float[Tensor, "pool features"],
        num_permutations: int = 1000,
        input_scale: float = 1,
        batch_size: int = 256,
        seed: Optional[int] = None,
    ):
        return self.combination_z_score(
            j,
            S,
            null_vectors,
            num_combinations=num_permutations,
            input_scale=input_scale,
            batch_size=batch_size,
            seed=seed,
        )

    def combination_z_score(
        self,
        j: int,
        S: Float[Tensor, "width features"],
        null_vectors: Float[Tensor, "pool features"],
        num_combinations: int = 100000,
        input_scale: float = 1,
        batch_size: int = 256,
        seed: Optional[int] = None,
    ):
        if S.ndim != 2 or null_vectors.ndim != 2:
            raise ValueError("S and null_vectors must both be two-dimensional")
        if S.shape[1] != null_vectors.shape[1]:
            raise ValueError(
                f"S dimension {S.shape[1]} does not match null vector dimension {null_vectors.shape[1]}"
            )
        if S.shape[0] > null_vectors.shape[0]:
            raise ValueError("the null pool must contain at least as many vectors as S")
        if num_combinations < 2:
            raise ValueError("num_combinations must be at least 2")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        generator = torch.Generator(device="cpu")
        if seed is not None:
            generator.manual_seed(seed)

        pool_size = null_vectors.shape[0]
        width = S.shape[0]
        total_combinations = math.comb(pool_size, width)
        num_combinations = min(num_combinations, total_combinations)
        null_scores = []
        if num_combinations == total_combinations:
            indices = torch.arange(pool_size, dtype=torch.long)
            combination_batches = _streamed_combinations(indices, width, batch_size)
        else:
            def sampled_combination_batches():
                for start in range(0, num_combinations, batch_size):
                    count = min(batch_size, num_combinations - start)
                    batch = torch.empty((0, width), dtype=torch.long)
                    while batch.shape[0] < count:
                        candidates = torch.randint(
                            pool_size,
                            (count - batch.shape[0], width),
                            generator=generator,
                        ).sort(dim=1).values
                        candidates = candidates[
                            torch.all(candidates[:, 1:] != candidates[:, :-1], dim=1)
                        ]
                        batch = torch.cat((batch, candidates))
                    yield batch

            combination_batches = sampled_combination_batches()

        for indices in combination_batches:
            indices = indices.to(null_vectors.device)
            scores = self.I_vectors(j, null_vectors[indices], input_scale)
            null_scores.append(scores.detach().float().cpu())

        observed = self.I_vectors(j, S, input_scale).detach().float().cpu()
        null_scores = torch.cat(null_scores)
        print(observed.max())
        print(null_scores.max())
        null_mean = null_scores.mean()
        null_std = null_scores.std(correction=0)
        if null_std == 0:
            raise ValueError("the permutation null has zero standard deviation")

        return {
            "z_score": (observed - null_mean) / null_std,
            "observed": observed,
            "null_mean": null_mean,
            "null_std": null_std,
            "null_mean_se": null_std / math.sqrt(num_combinations),
            "empirical_p_two_sided": (
                (null_scores.sub(null_mean).abs() >= observed.sub(null_mean).abs()).sum() + 1
            ) / (num_combinations + 1),
            "num_combinations": num_combinations,
            "total_combinations": total_combinations,
            "null_scores": null_scores,
        }

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