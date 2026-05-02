from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import torch
from torch import nn
import torch.nn.functional as F

from .memory import HebbianMemory


@dataclass
class TEMConfig:
    n_observations: int
    n_actions: int
    n_states: int
    g_dim: int = 64       # abstract/path-integration code, MEC-like
    x_dim: int = 32       # compressed sensory code, LEC-like
    p_dim: int = 128      # conjunctive hippocampal code
    hidden_dim: int = 128
    eta: float = 0.4
    lam: float = 0.995
    retrieve_iter: int = 3
    kappa: float = 0.8
    state_loss_weight: float = 1.0
    obs_loss_weight: float = 1.0
    memory_loss_weight: float = 0.1


class TEMLite(nn.Module):
    """A compact TEM-like model for aliased action-observation sequences.

    This model is intentionally simpler than the original TEM implementation:
    - `g_t` is updated by a GRUCell from previous `g` and action embedding.
    - `x_t` encodes the current observation.
    - `p_t` conjunctively binds `g_t` and `x_t`.
    - a Hebbian memory retrieves `p_t`-like patterns from prior experience.
    - decoders predict current observation and latent state.

    It can be trained with state supervision for diagnostics, while the core
    prediction objective remains sensory/action sequence learning.
    """

    def __init__(self, cfg: TEMConfig):
        super().__init__()
        self.cfg = cfg
        self.obs_emb = nn.Embedding(cfg.n_observations, cfg.x_dim)
        self.act_emb = nn.Embedding(cfg.n_actions, cfg.g_dim)
        self.g_init = nn.Parameter(torch.zeros(cfg.g_dim))
        self.path_integrator = nn.GRUCell(cfg.g_dim, cfg.g_dim)
        self.p_encoder = nn.Sequential(
            nn.Linear(cfg.g_dim + cfg.x_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.p_dim),
            nn.ReLU(),
        )
        self.mem_to_g = nn.Linear(cfg.p_dim, cfg.g_dim)
        self.obs_decoder = nn.Linear(cfg.p_dim, cfg.n_observations)
        self.state_decoder = nn.Linear(cfg.p_dim, cfg.n_states)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor, states: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Run a batch of sequences.

        Parameters
        ----------
        obs: LongTensor, shape (B, T)
            Observation symbols.
        actions: LongTensor, shape (B, T)
            Action taken at each step. The action at t is used to update g_{t+1};
            for simplicity this implementation predicts state/obs at t from obs_t
            and previous action history up to t-1.
        states: optional LongTensor, shape (B, T)
            Ground-truth latent position, used only for an auxiliary diagnostic loss.
        """
        B, T = obs.shape
        device = obs.device
        g = self.g_init.unsqueeze(0).expand(B, -1)
        mem = HebbianMemory(B, self.cfg.p_dim, device, self.cfg.eta, self.cfg.lam)

        obs_logits, state_logits, p_list, p_mem_list = [], [], [], []
        for t in range(T):
            if t > 0:
                a_prev = self.act_emb(actions[:, t - 1])
                g = self.path_integrator(a_prev, g)

            x = self.obs_emb(obs[:, t])
            p = self.p_encoder(torch.cat([g, x], dim=-1))
            p_norm = F.normalize(p + 1e-6, dim=-1)
            p_mem = mem.retrieve(p_norm, self.cfg.retrieve_iter, self.cfg.kappa)
            p_combined = F.normalize(p_norm + p_mem, dim=-1)

            obs_logits.append(self.obs_decoder(p_combined))
            state_logits.append(self.state_decoder(p_combined))
            p_list.append(p_norm)
            p_mem_list.append(p_mem)

            # Fast memory update is detached by design: it is an online episode memory.
            mem.update(p_norm)

        out = {
            "obs_logits": torch.stack(obs_logits, dim=1),
            "state_logits": torch.stack(state_logits, dim=1),
            "p": torch.stack(p_list, dim=1),
            "p_mem": torch.stack(p_mem_list, dim=1),
        }
        out["loss"] = self.loss(out, obs, states)
        return out

    def loss(self, out: Dict[str, torch.Tensor], obs: torch.Tensor, states: Optional[torch.Tensor]) -> torch.Tensor:
        cfg = self.cfg
        loss_obs = F.cross_entropy(out["obs_logits"].reshape(-1, cfg.n_observations), obs.reshape(-1))
        loss_mem = F.mse_loss(out["p_mem"][:, 1:], out["p"][:, :-1].detach()) if out["p"].shape[1] > 1 else obs.new_tensor(0.0, dtype=torch.float)
        loss = cfg.obs_loss_weight * loss_obs + cfg.memory_loss_weight * loss_mem
        if states is not None:
            loss_state = F.cross_entropy(out["state_logits"].reshape(-1, cfg.n_states), states.reshape(-1))
            loss = loss + cfg.state_loss_weight * loss_state
        return loss
