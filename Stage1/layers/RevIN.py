import torch
import torch.nn as nn

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=False, subtract_last=False):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

        self.mean = None
        self.stdev = None
        self.last = None

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def get_stats(self):

        if self.stdev is None:
            raise RuntimeError("RevIN stats not set: stdev is None")

        if self.subtract_last:
            if self.last is None:
                raise RuntimeError("RevIN stats not set: last is None")
            return {"last": self.last, "stdev": self.stdev}
        else:
            if self.mean is None:
                raise RuntimeError("RevIN stats not set: mean is None")
            return {"mean": self.mean, "stdev": self.stdev}

    def set_stats(self, stats: dict, ref_tensor: torch.Tensor = None):
        if stats is None:
            return

        def _cast(t):
            if ref_tensor is None or not torch.is_tensor(t):
                return t
            return t.to(device=ref_tensor.device, dtype=ref_tensor.dtype)

        if self.subtract_last:
            self.last = _cast(stats["last"])
            self.stdev = _cast(stats["stdev"])
        else:
            self.mean = _cast(stats["mean"])
            self.stdev = _cast(stats["stdev"])

    def forward(self, x, mode: str, stats: dict = None, return_stats: bool = False):
        """
        x: [B, L, C]
        mode: 'norm' or 'denorm'
        stats: decode 时显式传入
        return_stats: norm 时是否返回 (x_norm, stats)
        """
        if mode == "norm":
            self._get_statistics(x)
            x = self._normalize(x)
            if return_stats:
                return x, self.get_stats()
            return x

        elif mode == "denorm":
            if stats is not None:
                self.set_stats(stats, ref_tensor=x)
            if self.stdev is None or (self.subtract_last and self.last is None) or ((not self.subtract_last) and self.mean is None):
                raise RuntimeError("RevIN denorm requires stats (or a prior norm call in the same module instance).")
            x = self._denormalize(x)
            return x

        else:
            raise NotImplementedError

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1).detach()
            self.mean = None  
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
            self.last = None
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x
