from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Constants / conventions
# -----------------------------
NXBIN = 12
NYBIN = 24          # 12x24 = 288 bins total (your N_NODES=288)
N_POS = NXBIN * NYBIN

# ego tokens
SOS_TOKEN = 4
PAD_TOKEN = 5
N_TOKENS = 6        # {0,1,2,3,SOS,PAD}

# position tokens: 0..287 are valid positions, 288 is PAD
PAD_POS_TOKEN = N_POS
N_POS_TOKENS = N_POS + 1

IGNORE_INDEX = -100


# -----------------------------
# Helpers: (node id) <-> (x,y)
# -----------------------------
def node_to_xy(node: int) -> Tuple[int, int]:
    """1-based node id in [1..288] -> integer (x,y) with y in [0..23]."""
    if node < 1 or node > N_POS:
        raise ValueError(f"node must be in [1,{N_POS}], got {node}")
    x = (node - 1) % NXBIN
    y = (node - 1) // NXBIN
    return int(x), int(y)

def xy_to_pos_id(x: int, y: int) -> int:
    """(x,y) -> flattened position id in [0..287]."""
    if not (0 <= x < NXBIN and 0 <= y < NYBIN):
        raise ValueError(f"(x,y) out of bounds: {(x,y)}")
    return int(y * NXBIN + x)

def pos_id_to_xy(pos_id: int) -> Tuple[int, int]:
    """flattened id in [0..287] -> (x,y)."""
    if pos_id < 0 or pos_id >= N_POS:
        raise ValueError(f"pos_id must be in [0,{N_POS-1}], got {pos_id}")
    x = pos_id % NXBIN
    y = pos_id // NXBIN
    return int(x), int(y)

def nodes_to_pos_ids(nodes: np.ndarray) -> np.ndarray:
    """nodes: 1..288 -> pos_ids: 0..287"""
    nodes = np.asarray(nodes, dtype=np.int64)
    if np.any(nodes < 1) or np.any(nodes > N_POS):
        raise ValueError(f"Nodes must be in [1,{N_POS}]")
    return (nodes - 1).astype(np.int64)

def sanitize_ego_tokens(ego_actions: np.ndarray) -> np.ndarray:
    """
    Map ego actions to tokens:
      -1 -> SOS
      0..3 -> unchanged
      PAD_TOKEN allowed
    """
    a = np.asarray(ego_actions, dtype=np.int64)
    a = np.where(a == -1, SOS_TOKEN, a)

    allowed = {0, 1, 2, 3, SOS_TOKEN, PAD_TOKEN}
    ok = np.isin(a, list(allowed))
    if not np.all(ok):
        bad = a[~ok]
        raise ValueError(f"Found invalid ego tokens: {bad[:10]}")
    return a


# -----------------------------
# CA3 belief model with (x,y) cue input
# -----------------------------
class CA3XYBeliefAgent(nn.Module):
    """
    Inputs:
      - ego_tokens: (B,T) in {0,1,2,3,SOS,PAD}
      - pos_tokens: (B,T) in {0..287, PAD_POS_TOKEN}, where typically only t=0 is provided:
            pos_tokens[:,0] = pos_id(x0,y0)
            pos_tokens[:,1:] = PAD_POS_TOKEN

    Output:
      - logits over positions: (B,T,288)
        belief grid = softmax(logits).reshape(B,T,NYBIN,NXBIN)
    """

    def __init__(
        self,
        ego_embed_dim: int = 32,
        pos_embed_dim: int = 32,
        ca3_hidden_dim: int = 128,
        num_gru_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.ego_emb = nn.Embedding(N_TOKENS, ego_embed_dim)
        self.pos_emb = nn.Embedding(N_POS_TOKENS, pos_embed_dim)

        self.ca3 = nn.GRU(
            input_size=ego_embed_dim + pos_embed_dim,
            hidden_size=ca3_hidden_dim,
            num_layers=num_gru_layers,
            batch_first=True,
            dropout=(dropout if num_gru_layers > 1 else 0.0),
        )
        self.readout = nn.Linear(ca3_hidden_dim, N_POS)

    def forward(
        self,
        ego_tokens: torch.LongTensor,      # (B,T)
        pos_tokens: torch.LongTensor,      # (B,T)
        *,
        lengths: Optional[torch.LongTensor] = None,
        return_hidden: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        if ego_tokens.shape != pos_tokens.shape:
            raise ValueError(f"ego_tokens and pos_tokens must have same shape, got {ego_tokens.shape} vs {pos_tokens.shape}")
        if ego_tokens.dim() != 2:
            raise ValueError(f"tokens must be (B,T), got {ego_tokens.shape}")

        x_ego = self.ego_emb(ego_tokens)   # (B,T,E1)
        x_pos = self.pos_emb(pos_tokens)   # (B,T,E2)
        x = torch.cat([x_ego, x_pos], dim=-1)

        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            out_packed, hT = self.ca3(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True, total_length=x.shape[1])
        else:
            out, hT = self.ca3(x)          # out: (B,T,H)

        logits = self.readout(out)         # (B,T,288)
        if return_hidden:
            return logits, hT
        return logits

    def forward_with_latents(
        self,
        ego_tokens: torch.LongTensor,
        pos_tokens: torch.LongTensor,
        *,
        lengths: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return logits, h_seq, hT."""
        if ego_tokens.shape != pos_tokens.shape:
            raise ValueError("ego_tokens and pos_tokens must have same shape.")
        x = torch.cat([self.ego_emb(ego_tokens), self.pos_emb(pos_tokens)], dim=-1)

        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            out_packed, hT = self.ca3(packed)
            h_seq, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True, total_length=x.shape[1])
        else:
            h_seq, hT = self.ca3(x)

        logits = self.readout(h_seq)
        return logits, h_seq, hT

    @torch.no_grad()
    def predict_belief_xy(
        self,
        ego_actions: np.ndarray,
        *,
        cue_xy: Tuple[int, int] = (0, 0),
        device: Optional[torch.device] = None,
    ) -> np.ndarray:
        """
        Returns belief grid: (T, NYBIN, NXBIN).
        """
        self.eval()
        device = device or next(self.parameters()).device

        ego = sanitize_ego_tokens(ego_actions)
        T = len(ego)

        # pos_tokens: only provide cue at t=0
        x0, y0 = cue_xy
        pos0 = xy_to_pos_id(x0, y0)
        pos_tokens = np.full((T,), PAD_POS_TOKEN, dtype=np.int64)
        pos_tokens[0] = pos0

        ego_t = torch.as_tensor(ego[None, :], dtype=torch.long, device=device)
        pos_t = torch.as_tensor(pos_tokens[None, :], dtype=torch.long, device=device)

        logits = self.forward(ego_t, pos_t)        # (1,T,288)
        belief = torch.softmax(logits, dim=-1)     # (1,T,288)
        belief_grid = belief.view(1, T, NYBIN, NXBIN)
        return belief_grid.squeeze(0).cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def predict_map_xy(
        self,
        ego_actions: np.ndarray,
        *,
        cue_xy: Tuple[int, int] = (0, 0),
        device: Optional[torch.device] = None,
    ) -> np.ndarray:
        """
        Returns MAP (x,y) for each time step: shape (T,2), integers.
        """
        belief_grid = self.predict_belief_xy(ego_actions, cue_xy=cue_xy, device=device)  # (T,NY,NX)
        flat = belief_grid.reshape(len(belief_grid), -1)  # (T,288)
        pos_id = flat.argmax(axis=-1)                     # (T,)
        xy = np.stack([pos_id % NXBIN, pos_id // NXBIN], axis=-1).astype(np.int64)
        return xy


# -----------------------------
# Batching + loss
# -----------------------------
@dataclass
class XYBeliefBatch:
    ego_tokens: torch.LongTensor     # (B,T)
    pos_tokens: torch.LongTensor     # (B,T)  only t=0 is cue; rest PAD_POS_TOKEN
    targets: torch.LongTensor        # (B,T)  pos_id in 0..287, padded = IGNORE_INDEX
    mask: torch.BoolTensor           # (B,T)
    lengths: torch.LongTensor        # (B,)

def make_xy_belief_batch(
    ego_batch: List[np.ndarray],
    node_batch: List[np.ndarray],
    *,
    device: torch.device,
    provide_full_pos_inputs: bool = False,
) -> XYBeliefBatch:
    """
    - ego_batch[i]: tokens length T_i (should include -1 at t=0 as SOS)
    - node_batch[i]: node ids length T_i in 1..288 (or your concatenated scheme)
    Inputs:
      pos_tokens: either only cue at t=0 (default), or full positions if provide_full_pos_inputs=True
    Targets:
      flattened pos id (node-1), padded = IGNORE_INDEX
    """
    if len(ego_batch) != len(node_batch):
        raise ValueError("ego_batch and node_batch must have same length.")

    B = len(ego_batch)
    lengths = [len(e) for e in ego_batch]
    T_max = max(lengths)

    ego_tokens = torch.full((B, T_max), PAD_TOKEN, dtype=torch.long, device=device)
    pos_tokens = torch.full((B, T_max), PAD_POS_TOKEN, dtype=torch.long, device=device)
    targets = torch.full((B, T_max), IGNORE_INDEX, dtype=torch.long, device=device)
    mask = torch.zeros((B, T_max), dtype=torch.bool, device=device)

    for i, (ego, nodes) in enumerate(zip(ego_batch, node_batch)):
        ego = sanitize_ego_tokens(ego)
        nodes = np.asarray(nodes, dtype=np.int64)
        if len(ego) != len(nodes):
            raise ValueError(f"Sample {i} length mismatch: len(ego)={len(ego)} vs len(nodes)={len(nodes)}")

        Ti = len(ego)
        ego_tokens[i, :Ti] = torch.from_numpy(ego).to(device=device)

        # targets are true positions (pos_id = node-1)
        pos_ids = nodes_to_pos_ids(nodes)  # 0..287
        targets[i, :Ti] = torch.from_numpy(pos_ids).to(device=device)
        mask[i, :Ti] = True

        if provide_full_pos_inputs:
            # feed true position at every time step (usually NOT what you want for retrieval)
            pos_tokens[i, :Ti] = torch.from_numpy(pos_ids).to(device=device)
        else:
            # only provide the initial cue (x0,y0) at t=0
            pos_tokens[i, 0] = int(pos_ids[0])

    lengths_t = torch.tensor(lengths, dtype=torch.long, device=device)
    return XYBeliefBatch(
        ego_tokens=ego_tokens,
        pos_tokens=pos_tokens,
        targets=targets,
        mask=mask,
        lengths=lengths_t,
    )

def ce_loss_positions(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    logits: (B,T,288)
    targets: (B,T) with IGNORE_INDEX for padded positions
    """
    B, T, C = logits.shape
    return F.cross_entropy(
        logits.view(B * T, C),
        targets.view(B * T),
        ignore_index=IGNORE_INDEX,
    )


# -----------------------------
# Training loop
# -----------------------------
def train_xy_belief_agent(
    model: CA3XYBeliefAgent,
    ego_dataset: List[np.ndarray],
    node_dataset: List[np.ndarray],
    *,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    grad_clip: float = 1.0,
    device: Optional[torch.device] = None,
) -> None:
    if len(ego_dataset) != len(node_dataset):
        raise ValueError("ego_dataset and node_dataset must have same length.")

    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model.to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    N = len(ego_dataset)
    idx = np.arange(N)

    for ep in range(1, epochs + 1):
        np.random.shuffle(idx)
        losses = []

        for s in range(0, N, batch_size):
            batch_idx = idx[s : s + batch_size].tolist()
            ego_b = [ego_dataset[i] for i in batch_idx]
            nod_b = [node_dataset[i] for i in batch_idx]

            batch = make_xy_belief_batch(ego_b, nod_b, device=device, provide_full_pos_inputs=False)

            logits = model(batch.ego_tokens, batch.pos_tokens, lengths=batch.lengths)  # (B,T,288)
            loss = ce_loss_positions(logits, batch.targets)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip and grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

            losses.append(float(loss.detach().cpu().item()))

        print(f"epoch {ep:03d} | loss={np.mean(losses):.6f}")


# -----------------------------
# Retrieval with latents
# -----------------------------
@torch.no_grad()
def retrieve_with_latents_xy(
    model: CA3XYBeliefAgent,
    ego_actions: np.ndarray,
    *,
    cue_xy: Tuple[int, int] = (0, 0),
    device: Optional[torch.device] = None,
):
    """
    Returns:
      belief_grid: (T, NYBIN, NXBIN)
      h_seq:       (T, H)
      hT:          (L, 1, H)
    """
    model.eval()
    device = device or next(model.parameters()).device

    ego = sanitize_ego_tokens(ego_actions)
    T = len(ego)

    pos_tokens = np.full((T,), PAD_POS_TOKEN, dtype=np.int64)
    pos_tokens[0] = xy_to_pos_id(*cue_xy)

    ego_t = torch.as_tensor(ego[None, :], dtype=torch.long, device=device)
    pos_t = torch.as_tensor(pos_tokens[None, :], dtype=torch.long, device=device)

    logits, h_seq, hT = model.forward_with_latents(ego_t, pos_t)
    belief = torch.softmax(logits, dim=-1).view(1, T, NYBIN, NXBIN)

    return (
        belief.squeeze(0).cpu().numpy().astype(np.float32),
        h_seq.squeeze(0).cpu().numpy().astype(np.float32),
        hT.cpu().numpy(),
    )