import gc
import math

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

    def I(self, j: int, width: int, batch_size: int, k: int, input_scale: int = 1):
        device = self.device

        try:
            top_k, bottom_k = StreamedTopK(top_k=k), StreamedTopK(top_k=k, mode="bottom")
            x = torch.arange(self.V.shape[1], device=device, dtype=torch.int16)
            S = _streamed_combinations(x, width, batch_size)
            S_len = math.comb(self.V.shape[1], width)

            with tqdm(total=S_len) as pbar:
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