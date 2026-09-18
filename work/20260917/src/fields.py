"""Only the two requested graph fields are new; never allocate dense G x G."""
import torch
from torch import nn
from torch.nn import functional as F
from ODE.ode_20260609_mathmlp import _mlp, _prep_t


class SparseField(nn.Module):
    soft = True

    def off_mask_penalty(self, norm="l1"):
        # Off-support parameters do not exist. The inherited off-mask loss is zero.
        return next(self.parameters()).new_zeros(())


class DirectMessageODE(SparseField):
    def __init__(self, d, src, dst, hidden=64, chunk_size=4096):
        super().__init__()
        self.d, self.chunk_size = d, chunk_size
        self.register_buffer("src", src.clone())
        self.register_buffer("dst", dst.clone())
        self.weight = nn.Parameter(torch.ones(len(src)) / max(d, 1)**.5)
        self.gamma = nn.Parameter(torch.full((d,), .1))
        self.phi = _mlp(3, hidden, 1)

    def forward(self, x, t=None):
        x = x.float()
        times = _prep_t(t, len(x), x.device, x.dtype)[:, None]
        result = -F.softplus(self.gamma) * x
        # One shared phi; vectorized over batch and a bounded edge chunk.
        for start in range(0, len(self.src), self.chunk_size):
            src, dst = self.src[start:start+self.chunk_size], self.dst[start:start+self.chunk_size]
            inputs = torch.stack((x[:, dst], x[:, src], times.expand(-1, len(src))), -1)
            message = self.phi(inputs).squeeze(-1) * self.weight[start:start+self.chunk_size]
            result = result.index_add(1, dst, message)
        return result


class MultiHopGraphFilterODE(SparseField):
    def __init__(self, d, src, dst, hidden=64):
        super().__init__()
        self.d = d
        self.register_buffer("src", src.clone())
        self.register_buffer("dst", dst.clone())
        degree = torch.bincount(dst, minlength=d).float().clamp_min(1)
        # Store COO components, not a sparse state_dict tensor (EMA remains native).
        self.register_buffer("p_values", 1 / degree[dst])
        self.f_theta = _mlp(5, hidden, 1)

    def forward(self, x, t=None):
        x = x.float()
        p = torch.sparse_coo_tensor(torch.stack((self.dst, self.src)), self.p_values,
                                    (self.d, self.d), device=x.device).coalesce()
        h1 = torch.sparse.mm(p, x.T).T
        h2 = torch.sparse.mm(p, h1.T).T
        h3 = torch.sparse.mm(p, h2.T).T
        times = _prep_t(t, len(x), x.device, x.dtype)[:, None].expand_as(x)
        return self.f_theta(torch.stack((x, h1, h2, h3, times), -1)).squeeze(-1)
