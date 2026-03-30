from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


N_POS = 144
N_OBS = 6
PAD_OBS_TOKEN = 6
N_OBS_TOKENS = 7
IGNORE_INDEX = -100


class CA3ActionBeliefAgent(nn.Module):
    """
    Same architecture for different action spaces.

    Inputs:
      - obs_tokens: (B,T)
      - prev_action_tokens: (B,T)

    Outputs:
      - pos_logits: (B,T,N_POS)
      - act_logits: (B,T,n_actions_predict)
      - obs_logits: (B,T,N_OBS), optional
    """

    def __init__(
        self,
        n_action_tokens: int,
        n_action_classes: int,
        obs_embed_dim: int = 16,
        act_embed_dim: int = 16,
        hidden_dim: int = 128,
        num_gru_layers: int = 1,
        dropout: float = 0.0,
        predict_next_obs: bool = True,
    ) -> None:
        super().__init__()

        self.predict_next_obs = predict_next_obs
        self.n_action_tokens = n_action_tokens
        self.n_action_classes = n_action_classes

        self.obs_emb = nn.Embedding(N_OBS_TOKENS, obs_embed_dim)
        self.act_emb = nn.Embedding(n_action_tokens, act_embed_dim)

        self.ca3 = nn.GRU(
            input_size=obs_embed_dim + act_embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_gru_layers,
            batch_first=True,
            dropout=(dropout if num_gru_layers > 1 else 0.0),
        )

        self.pos_head = nn.Linear(hidden_dim, N_POS)
        self.act_head = nn.Linear(hidden_dim, n_action_classes)
        self.obs_head = nn.Linear(hidden_dim, N_OBS) if predict_next_obs else None

    def forward(
        self,
        obs_tokens: torch.LongTensor,
        prev_action_tokens: torch.LongTensor,
        *,
        lengths: Optional[torch.LongTensor] = None,
        return_hidden: bool = False,
    ):
        if obs_tokens.shape != prev_action_tokens.shape:
            raise ValueError("obs_tokens and prev_action_tokens must have same shape.")

        x = torch.cat(
            [self.obs_emb(obs_tokens), self.act_emb(prev_action_tokens)],
            dim=-1,
        )

        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            out_packed, hT = self.ca3(packed)
            h_seq, _ = nn.utils.rnn.pad_packed_sequence(
                out_packed, batch_first=True, total_length=x.shape[1]
            )
        else:
            h_seq, hT = self.ca3(x)

        pos_logits = self.pos_head(h_seq)
        act_logits = self.act_head(h_seq)
        obs_logits = self.obs_head(h_seq) if self.obs_head is not None else None

        if return_hidden:
            return pos_logits, act_logits, obs_logits, h_seq, hT
        return pos_logits, act_logits, obs_logits
