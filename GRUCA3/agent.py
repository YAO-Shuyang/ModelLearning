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
NYBIN = 12
N_NODES = 288

# Ego-actions:
#  0: Left, 1: Forward, 2: Right, 3: Backward
# We will map:
#  -1 -> SOS token
#  0..3 -> actions
#  PAD -> padding for batching
SOS_TOKEN = 4
PAD_TOKEN = 5
N_TOKENS = 7  # {0,1,2,3,SOS,PAD}


def sanitize_ego_tokens(ego_actions: np.ndarray) -> np.ndarray:
    """
    Convert ego-action array into token ids in {0,1,2,3,SOS,PAD}.

    Rules:
      -1 -> SOS_TOKEN
      0..3 -> unchanged
      PAD_TOKEN is allowed (for already-padded sequences)
    """
    a = np.asarray(ego_actions, dtype=np.int64)
    a = np.where(a == -1, SOS_TOKEN, a)

    allowed = {0, 1, 2, 3, 6, SOS_TOKEN, PAD_TOKEN}
    # vectorized membership check
    ok = np.isin(a, list(allowed))
    if not np.all(ok):
        bad = a[~ok]
        raise ValueError(f"Found invalid ego tokens: {bad[:10]}")
    return a

def check_nodes(nodes: np.ndarray) -> np.ndarray:
    nodes = np.asarray(nodes, dtype=np.int64)
    if np.any(nodes < 1) or np.any(nodes > N_NODES*2):
        raise ValueError("Nodes must be in [1,288].")
    return nodes


# -----------------------------
# CA3 Belief Model
# -----------------------------
class CA3BeliefAgent(nn.Module):
    """
    CA3-like recurrent belief model:
      input tokens: (B,T) in {0,1,2,3,SOS,PAD}
      recurrent core: GRU
      output: node logits (B,T,144) => belief = softmax over nodes

    Notes:
    - We do NOT need explicit heading or start node for training if your training
      always uses the same start cue token at t=0 (SOS).
    - For retrieval mismatch experiments, we can "mis-cue" by swapping the SOS embedding
      to a different learned cue embedding (see cue_id argument).
    """

    def __init__(
        self,
        token_embed_dim: int = 32,
        ca3_hidden_dim: int = 128,
        num_gru_layers: int = 1,
        dropout: float = 0.0,
        n_cues: int = 1,
    ) -> None:
        """
        n_cues:
          number of distinct start-cue embeddings you want available.
          If you set n_cues > 1, you can do mismatch retrieval by choosing cue_id != 0 at test time.
          Training typically uses cue_id=0 always.
        """
        super().__init__()
        self.token_emb = nn.Embedding(N_TOKENS, token_embed_dim)

        # Optional learned cue embeddings that can be injected at t=0
        # (instead of using only SOS_TOKEN embedding).
        self.n_cues = int(n_cues)
        self.cue_emb = nn.Embedding(self.n_cues, token_embed_dim)

        self.ca3 = nn.GRU(
            input_size=token_embed_dim,
            hidden_size=ca3_hidden_dim,
            num_layers=num_gru_layers,
            batch_first=True,
            dropout=(dropout if num_gru_layers > 1 else 0.0),
        )

        self.readout = nn.Linear(ca3_hidden_dim, N_NODES)

        self.num_gru_layers = num_gru_layers
        self.ca3_hidden_dim = ca3_hidden_dim

    def forward(
        self,
        tokens: torch.LongTensor,         # (B,T)
        *,
        cue_id: int = 0,
        lengths: Optional[torch.LongTensor] = None,  # (B,) optional
        return_hidden: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        tokens include SOS at t=0 ideally.
        If cue_id is used, we override the embedding at t=0 with a learned cue embedding.
        """
        if tokens.dim() != 2:
            raise ValueError(f"tokens must be (B,T), got {tokens.shape}")
        B, T = tokens.shape
        if not (0 <= cue_id < self.n_cues):
            raise ValueError(f"cue_id must be in [0,{self.n_cues-1}]")

        x = self.token_emb(tokens)  # (B,T,E)

        # Override the first time step embedding with cue embedding (retrieval cue).
        # This is the "handle" you can use later to mismatch.
        cue_vec = self.cue_emb(torch.tensor([cue_id], device=tokens.device)).view(1, 1, -1)  # (1,1,E)
        x[:, 0:1, :] = cue_vec.expand(B, 1, -1)

        # If you want to ignore PAD tokens properly, use packing when lengths is provided.
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            out_packed, hT = self.ca3(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True, total_length=T)
        else:
            out, hT = self.ca3(x)  # out: (B,T,H)

        logits = self.readout(out)  # (B,T,144)
        if return_hidden:
            return logits, hT
        return logits

    def forward_with_latents(
        self,
        tokens: torch.LongTensor,         # (B,T)
        *,
        cue_id: int = 0,
        lengths: torch.LongTensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        logits : (B,T,144)
        h_seq  : (B,T,H)   hidden state at each time step (latent trajectory)
        hT     : (L,B,H)   final hidden state (per layer)
        """
        B, T = tokens.shape

        x = self.token_emb(tokens)  # (B,T,E)

        # cue override (only if you kept cues)
        if getattr(self, "n_cues", 0) > 0:
            if not (0 <= cue_id < self.n_cues):
                raise ValueError(f"cue_id must be in [0,{self.n_cues-1}]")
            cue_vec = self.cue_emb(torch.tensor([cue_id], device=tokens.device)).view(1, 1, -1)
            x[:, 0:1, :] = cue_vec.expand(B, 1, -1)

        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            out_packed, hT = self.ca3(packed)
            h_seq, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True, total_length=T)
        else:
            h_seq, hT = self.ca3(x)  # h_seq: (B,T,H)

        logits = self.readout(h_seq)  # (B,T,144)
        return logits, h_seq, hT

    @torch.no_grad()
    def predict_belief(
        self,
        ego_actions: np.ndarray,
        *,
        cue_id: int = 0,
        device: Optional[torch.device] = None,
    ) -> np.ndarray:
        """
        Returns belief over nodes at each time step: shape (T,144).
        """
        self.eval()
        device = device or next(self.parameters()).device

        tokens = sanitize_ego_tokens(ego_actions)
        tokens_t = torch.as_tensor(tokens[None, :], dtype=torch.long, device=device)  # (1,T)
        logits = self.forward(tokens_t, cue_id=cue_id)  # (1,T,144)
        belief = torch.softmax(logits, dim=-1)
        return belief.squeeze(0).cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def predict_map_nodes(
        self,
        ego_actions: np.ndarray,
        *,
        cue_id: int = 0,
        device: Optional[torch.device] = None,
    ) -> np.ndarray:
        """
        Returns MAP node id (1..144) at each time step: shape (T,).
        """
        belief = self.predict_belief(ego_actions, cue_id=cue_id, device=device)  # (T,144)
        map_idx0 = belief.argmax(axis=-1)  # 0..143
        return (map_idx0 + 1).astype(np.int64)

@torch.no_grad()
def retrieve_with_latents(
    model: CA3BeliefAgent,
    ego_actions: np.ndarray,
    *,
    cue_id: int = 0,
    device: torch.device | None = None,
):
    model.eval()
    device = device or next(model.parameters()).device

    tokens = sanitize_ego_tokens(ego_actions)  # your function
    tokens_t = torch.as_tensor(tokens[None, :], dtype=torch.long, device=device)  # (1,T)
    
    logits, h_seq, hT = model.forward_with_latents(tokens_t, cue_id=cue_id)

    belief = torch.softmax(logits, dim=-1)  # (1,T,144)
    return (
        belief.squeeze(0).cpu().numpy(),   # (T,144)
        h_seq.squeeze(0).cpu().numpy(),    # (T,H)   <-- latent state over time
        hT.cpu().numpy(),                  # (L,1,H) final per-layer state
    )

# -----------------------------
# Batching + loss
# -----------------------------
@dataclass
class BeliefBatch:
    tokens: torch.LongTensor   # (B,T) in {0..3,SOS,PAD}
    targets: torch.LongTensor  # (B,T) in {0..143}  (node-1)
    mask: torch.BoolTensor     # (B,T) valid positions
    lengths: torch.LongTensor  # (B,) lengths (for packing)


def make_belief_batch(
    ego_batch: List[np.ndarray],
    node_batch: List[np.ndarray],
    *,
    device: torch.device,
) -> BeliefBatch:
    """
    Build padded batch.

    REQUIREMENT (recommended convention):
      - tokens length equals nodes length
      - first token corresponds to first node label
      - first token should be -1 (SOS) in your data (or you can insert it upstream)

    We pad tokens with PAD_TOKEN and targets with 0 (ignored via mask).
    """
    if len(ego_batch) != len(node_batch):
        raise ValueError("ego_batch and node_batch must have same length.")
    B = len(ego_batch)
    lengths = [len(e) for e in ego_batch]
    T_max = max(lengths)

    tokens = torch.full((B, T_max), PAD_TOKEN, dtype=torch.long, device=device)
    targets = torch.zeros((B, T_max), dtype=torch.long, device=device)
    mask = torch.zeros((B, T_max), dtype=torch.bool, device=device)

    for i, (ego, nodes) in enumerate(zip(ego_batch, node_batch)):
        ego = sanitize_ego_tokens(ego)
        nodes = check_nodes(nodes)
        if len(ego) != len(nodes):
            raise ValueError(
                f"Sample {i} length mismatch: len(ego)={len(ego)} vs len(nodes)={len(nodes)}. "
                "Make them aligned (same T)."
            )

        Ti = len(ego)
        tokens[i, :Ti] = torch.from_numpy(ego).to(device=device)
        targets[i, :Ti] = torch.from_numpy((nodes - 1).astype(np.int64)).to(device=device)
        mask[i, :Ti] = True

    lengths_t = torch.tensor(lengths, dtype=torch.long, device=device)
    return BeliefBatch(tokens=tokens, targets=targets, mask=mask, lengths=lengths_t)


def masked_cross_entropy(
    logits: torch.Tensor,        # (B,T,144)
    targets: torch.Tensor,       # (B,T) in 0..143
    mask: torch.BoolTensor,      # (B,T)
) -> torch.Tensor:
    B, T, C = logits.shape
    logits2 = logits.reshape(B * T, C)
    targets2 = targets.reshape(B * T)
    mask2 = mask.reshape(B * T)

    # standard CE per position, then mask
    loss_all = F.cross_entropy(logits2, targets2, reduction="none")  # (B*T,)
    loss = loss_all[mask2].mean()
    return loss


# -----------------------------
# Training loop
# -----------------------------
def train_belief_agent(
    model: CA3BeliefAgent,
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
    """
    Supervised training:
      input: ego tokens
      target: true node id at each time step
      loss: masked cross-entropy
    """
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

            batch = make_belief_batch(ego_b, nod_b, device=device)

            logits = model(batch.tokens, cue_id=0, lengths=batch.lengths)  # (B,T,144)
            loss = masked_cross_entropy(logits, batch.targets, batch.mask)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip and grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

            losses.append(float(loss.detach().cpu().item()))

        print(f"epoch {ep:03d} | loss={np.mean(losses):.6f}")


# -----------------------------
# Example: retrieval mismatch demo
# -----------------------------
@torch.no_grad()
def retrieval_mismatch_demo(
    model: CA3BeliefAgent,
    ego_actions: np.ndarray,
    true_nodes: np.ndarray,
    *,
    cue_id_test: int,
    device: Optional[torch.device] = None,
) -> None:
    """
    Show how belief behaves when you use a mismatched cue_id.
    Prints:
      - MAP node at a few time points
      - probability assigned to true node over time (summary)
    """
    device = device or next(model.parameters()).device
    belief = model.predict_belief(ego_actions, cue_id=cue_id_test, device=device)  # (T,144)
    map_nodes = belief.argmax(axis=-1) + 1  # (T,)

    true_nodes = check_nodes(true_nodes)
    true_idx0 = (true_nodes - 1).astype(np.int64)
    p_true = belief[np.arange(len(true_nodes)), true_idx0]

    print("time | MAP_node | true_node | p(true)")
    for t in np.linspace(0, len(true_nodes) - 1, num=min(10, len(true_nodes)), dtype=int):
        print(f"{t:4d} | {map_nodes[t]:8d} | {true_nodes[t]:9d} | {p_true[t]:.3f}")

    print(f"\nmean p(true) = {float(np.mean(p_true)):.3f}")
    print(f"final p(true) = {float(p_true[-1]):.3f}")


if __name__ == "__main__":
    from rtrv_models.MazeExplorer.graph import MazeEnv, maze1_graph

    # Example trajectory nodes (start fixed at node 1)
    train_nodes = np.array([1,13,14,26,27,15,3,4,5,6,18,17,29,30,31,19,20,21,9,10,11,12,24,23,22,34,33,32,44,45,46,47,48,60,59,58,57,56,68,69,70,71,72,84,83,95,94,82,81,80,92,104,103,91,90,78,79,67,55,54,66,65,64,63,75,74,62,50,51,39,38,37,49,61,73,85,97,109,110,122,123,111,112,100,99,87,88,76,77,89,101,102,114,113,125,124,136,137,138,126,127,115,116,117,129,141,142,130,131,132,144],dtype = np.int64)
    maze1 = MazeEnv(graph=maze1_graph, start_node=1, goal_node=144)
    train_ego = maze1.to_ego_actions(train_nodes)
    
    train_nodes_m2 = np.array([1,2,14,13,25,37,49,61,73,74,86,85,97,98,110,109,121,133,134,135,123,111,99,100,88,76,75,63,64,65,53,52,40,28,29,17,5,6,7,19,18,30,31,43,44,32,33,21,9,10,11,12,24,23,22,34,46,45,57,69,68,67,55,54,66,78,77,89,101,102,114,115,116,104,103,91,79,80,92,93,105,106,94,82,70,71,59,60,72,84,83,95,96,108,120,119,131,130,142,143,144], dtype = np.int64)
    train_ego_m2 = maze1.to_ego_actions(train_nodes_m2)
    
    train_nodes = np.concatenate([train_nodes, train_nodes_m2+144])
    train_ego = np.concatenate([train_ego, train_ego_m2])
    
    retrv_nodes = np.array([99,87,88,76,77,89,101,102,114,113,125,124,136,137,138,126,127,115,116,117,129,141,142,130,131,132,144], np.int64)
    retrv_ego = maze1.to_ego_actions(retrv_nodes)

    model = CA3BeliefAgent(token_embed_dim=32, ca3_hidden_dim=128, n_cues=1)

    belief, h_seq, hT = retrieve_with_latents(model, retrv_ego, cue_id=0)
    print(belief.shape)  # (T,144)
    print(h_seq.shape)   # (T,H)
    print(hT.shape)      # (L,1,H)

    # Train on a tiny dataset
    train_belief_agent(model, ego_dataset=[train_ego]*50, node_dataset=[train_nodes]*50, epochs=200, batch_size=128)

    # Normal cue (cue_id=0)
    print("\nNormal cue (cue_id=0):")
    retrieval_mismatch_demo(model, train_ego, train_nodes, cue_id_test=0)

    # Mismatched cue (cue_id=1) -- this is your "wrong initialization" handle
    print("\nMismatched cue (cue_id=1):")
    retrieval_mismatch_demo(model, retrv_ego, retrv_nodes, cue_id_test=0)