from __future__ import annotations

import torch


@torch.no_grad()
def decode_position(model, obs, actions):
    """Return posterior-like softmax over latent states for each time point."""
    out = model(obs, actions, states=None)
    return torch.softmax(out["state_logits"], dim=-1)
