from __future__ import annotations

import torch
import torch.nn.functional as F


class HebbianMemory:
    """Fast within-episode memory M updated by outer products of p codes."""

    def __init__(self, batch_size: int, dim: int, device: torch.device, eta: float = 0.5, lam: float = 0.995):
        self.M = torch.zeros(batch_size, dim, dim, device=device)
        self.eta = float(eta)
        self.lam = float(lam)

    def retrieve(self, cue: torch.Tensor, n_iter: int = 3, kappa: float = 0.8) -> torch.Tensor:
        h = cue
        for _ in range(n_iter):
            h_new = torch.bmm(h.unsqueeze(1), self.M).squeeze(1)
            h = F.normalize(kappa * h + (1.0 - kappa) * h_new, dim=-1)
        return h

    def update(self, key: torch.Tensor, value: torch.Tensor | None = None) -> None:
        if value is None:
            value = key
        outer = torch.bmm(key.unsqueeze(2), value.unsqueeze(1))
        self.M = self.lam * self.M + self.eta * outer.detach()
