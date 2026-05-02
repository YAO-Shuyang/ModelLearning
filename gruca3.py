from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from rtrv_models.base import BaseModel


# =========================
# Constants
# =========================
N_POS = 144

N_OBS = 6
PAD_OBS_TOKEN = 6
N_OBS_TOKENS = 7
IGNORE_INDEX = -100


# =========================
# Low-level GRU network
# =========================
class GRUCA3Net(nn.Module):
    """
    GRU-based sequence model.

    Inputs
    ------
    obs_tokens : (B, T)
        Observation tokens.
    prev_action_tokens : (B, T)
        Previous-action tokens, aligned to obs_tokens.
    lengths : Optional[(B,)]
        Valid lengths before padding.

    Outputs
    -------
    pos_logits : (B, T, N_POS)
    act_logits : (B, T, n_action_classes)
    obs_logits : (B, T, N_OBS) or None
        If present, meant to predict next observation.
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


# =========================
# Utilities
# =========================
def _ensure_int64(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.int64)


def _check_same_length(*arrays: np.ndarray) -> None:
    lengths = [arr.shape[0] for arr in arrays if arr is not None]
    if len(set(lengths)) > 1:
        raise ValueError(f"Arrays must have the same length, got lengths={lengths}.")


def _validate_trial_bounds(
    trial_beg: np.ndarray,
    trial_end: np.ndarray,
    n_total: int,
) -> None:
    if trial_beg.shape != trial_end.shape:
        raise ValueError("trial_beg and trial_end must have the same shape.")
    if np.any(trial_beg < 0) or np.any(trial_end < 0):
        raise ValueError("trial bounds must be non-negative.")
    if np.any(trial_beg >= trial_end):
        raise ValueError("Each trial must satisfy trial_beg < trial_end.")
    if np.max(trial_end) > n_total:
        raise ValueError("trial_end exceeds data length.")
    if np.any(np.diff(trial_beg) < 0) or np.any(np.diff(trial_end) < 0):
        raise ValueError("trial_beg and trial_end should be sorted in ascending order.")


def _make_prev_action_tokens(
    act_seq: np.ndarray,
    start_action_token: int,
) -> np.ndarray:
    """
    Build previous-action tokens for one trial:
    prev_act[0] = start_action_token
    prev_act[t] = act_seq[t-1] for t >= 1
    """
    prev = np.empty_like(act_seq)
    prev[0] = start_action_token
    if len(act_seq) > 1:
        prev[1:] = act_seq[:-1]
    return prev


# =========================
# Dataset for variable-length trials
# =========================
class _TrialDataset(Dataset):
    def __init__(
        self,
        obs_list: list[np.ndarray],
        prev_act_list: list[np.ndarray],
        pos_list: Optional[list[np.ndarray]],
        act_list: list[np.ndarray],
        next_obs_list: Optional[list[np.ndarray]],
    ) -> None:
        self.obs_list = obs_list
        self.prev_act_list = prev_act_list
        self.pos_list = pos_list
        self.act_list = act_list
        self.next_obs_list = next_obs_list

    def __len__(self) -> int:
        return len(self.obs_list)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        item = {
            "obs": self.obs_list[idx],
            "prev_act": self.prev_act_list[idx],
            "act": self.act_list[idx],
        }
        if self.pos_list is not None:
            item["pos"] = self.pos_list[idx]
        if self.next_obs_list is not None:
            item["next_obs"] = self.next_obs_list[idx]
        return item


def _collate_batch(batch: list[dict[str, np.ndarray]]) -> dict[str, torch.Tensor]:
    batch_size = len(batch)
    lengths = np.array([len(x["obs"]) for x in batch], dtype=np.int64)
    max_len = int(lengths.max())

    obs = np.full((batch_size, max_len), PAD_OBS_TOKEN, dtype=np.int64)
    prev_act = np.zeros((batch_size, max_len), dtype=np.int64)
    act = np.full((batch_size, max_len), IGNORE_INDEX, dtype=np.int64)
    pos = None
    next_obs = None

    has_pos = "pos" in batch[0]
    has_next_obs = "next_obs" in batch[0]

    if has_pos:
        pos = np.full((batch_size, max_len), IGNORE_INDEX, dtype=np.int64)
    if has_next_obs:
        next_obs = np.full((batch_size, max_len), IGNORE_INDEX, dtype=np.int64)

    for i, item in enumerate(batch):
        L = len(item["obs"])
        obs[i, :L] = item["obs"]
        prev_act[i, :L] = item["prev_act"]
        act[i, :L] = item["act"]
        if has_pos:
            pos[i, :L] = item["pos"]
        if has_next_obs:
            next_obs[i, :L] = item["next_obs"]

    out = {
        "obs": torch.as_tensor(obs, dtype=torch.long),
        "prev_act": torch.as_tensor(prev_act, dtype=torch.long),
        "act": torch.as_tensor(act, dtype=torch.long),
        "lengths": torch.as_tensor(lengths, dtype=torch.long),
    }
    if has_pos:
        out["pos"] = torch.as_tensor(pos, dtype=torch.long)
    if has_next_obs:
        out["next_obs"] = torch.as_tensor(next_obs, dtype=torch.long)
    return out


# =========================
# Wrapper class mimicking CSCG
# =========================
@dataclass
class GRUCA3(BaseModel):
    """
    A wrapper around a GRU-based CA3-like sequence model for convenient
    training, decoding, and comparison with other retrieval models.

    Parameters
    ----------
    act : np.ndarray[np.int64]
        Action sequence.
    obs : np.ndarray[np.int64]
        Observation sequence.
    pos : Optional[np.ndarray[np.int64]]
        Position / latent-state labels, if supervised training is desired.
        Expected to be 1..N_POS or 0..N_POS-1; internally converted to 0-based.
    n_action_classes : int
        Number of action classes to predict.
    n_action_tokens : Optional[int]
        Number of input action tokens. If None, set automatically as
        max(act) + 2 so one extra start token is available.
    model : Optional[GRUCA3Net]
        Underlying neural network.
    """

    act: np.ndarray
    obs: np.ndarray
    pos: Optional[np.ndarray] = None

    n_action_classes: int = 6
    n_action_tokens: Optional[int] = None

    obs_embed_dim: int = 16
    act_embed_dim: int = 16
    hidden_dim: int = 128
    num_gru_layers: int = 1
    dropout: float = 0.0
    predict_next_obs: bool = True

    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 32
    device: str = "cuda" if torch.cuda.is_available() else "cpu",

    pos_loss_weight: float = 1.0
    act_loss_weight: float = 1.0
    obs_loss_weight: float = 1.0

    model: Optional[GRUCA3Net] = None

    def __post_init__(self) -> None:
        self.act = _ensure_int64(self.act)
        self.obs = _ensure_int64(self.obs)
        if self.pos is not None:
            self.pos = _ensure_int64(self.pos)

        _check_same_length(self.act, self.obs, self.pos)

        if self.n_action_tokens is None:
            # Reserve one extra token for the start-of-trial previous action.
            self.n_action_tokens = int(np.max(self.act)) + 2

        self.start_action_token = self.n_action_tokens - 1

        if self.model is None:
            self.model = GRUCA3Net(
                n_action_tokens=self.n_action_tokens,
                n_action_classes=self.n_action_classes,
                obs_embed_dim=self.obs_embed_dim,
                act_embed_dim=self.act_embed_dim,
                hidden_dim=self.hidden_dim,
                num_gru_layers=self.num_gru_layers,
                dropout=self.dropout,
                predict_next_obs=self.predict_next_obs,
            )

        self.device = str(self.device)
        self.model.to(self.device)

    # -------------------------
    # internal helpers
    # -------------------------
    def _to_zero_based_pos(self, pos_seq: np.ndarray) -> np.ndarray:
        """
        Accept either 1..N_POS or 0..N_POS-1.
        Return 0-based position labels.
        """
        pos_seq = _ensure_int64(pos_seq)
        if pos_seq.size == 0:
            return pos_seq

        pos_min = int(pos_seq.min())
        pos_max = int(pos_seq.max())

        if pos_min >= 1 and pos_max <= N_POS:
            return pos_seq - 1
        if pos_min >= 0 and pos_max < N_POS:
            return pos_seq
        raise ValueError(
            f"Position labels must be in 1..{N_POS} or 0..{N_POS - 1}, "
            f"got min={pos_min}, max={pos_max}."
        )

    def _split_trials(
        self,
        obs_arr: np.ndarray,
        act_arr: np.ndarray,
        pos_arr: Optional[np.ndarray] = None,
        trial_beg: Optional[np.ndarray] = None,
        trial_end: Optional[np.ndarray] = None,
    ) -> tuple[list[np.ndarray], list[np.ndarray], Optional[list[np.ndarray]]]:
        """
        If trial bounds are None, treat the whole sequence as one trial.
        """
        obs_arr = _ensure_int64(obs_arr)
        act_arr = _ensure_int64(act_arr)
        pos_list: Optional[list[np.ndarray]] = None

        if pos_arr is not None:
            pos_arr = self._to_zero_based_pos(pos_arr)

        if trial_beg is None or trial_end is None:
            obs_list = [obs_arr]
            act_list = [act_arr]
            pos_list = [pos_arr] if pos_arr is not None else None
            return obs_list, act_list, pos_list

        trial_beg = _ensure_int64(trial_beg)
        trial_end = _ensure_int64(trial_end)
        _validate_trial_bounds(trial_beg, trial_end, len(obs_arr))

        obs_list = [obs_arr[b:e] for b, e in zip(trial_beg, trial_end)]
        act_list = [act_arr[b:e] for b, e in zip(trial_beg, trial_end)]
        if pos_arr is not None:
            pos_list = [pos_arr[b:e] for b, e in zip(trial_beg, trial_end)]
        return obs_list, act_list, pos_list

    def _build_dataset(
        self,
        obs_arr: np.ndarray,
        act_arr: np.ndarray,
        pos_arr: Optional[np.ndarray] = None,
        trial_beg: Optional[np.ndarray] = None,
        trial_end: Optional[np.ndarray] = None,
    ) -> _TrialDataset:
        obs_list, act_list, pos_list = self._split_trials(
            obs_arr=obs_arr,
            act_arr=act_arr,
            pos_arr=pos_arr,
            trial_beg=trial_beg,
            trial_end=trial_end,
        )

        prev_act_list = [
            _make_prev_action_tokens(act_seq, self.start_action_token)
            for act_seq in act_list
        ]

        next_obs_list = None
        if self.predict_next_obs:
            next_obs_list = []
            for obs_seq in obs_list:
                next_obs = np.full_like(obs_seq, IGNORE_INDEX)
                if len(obs_seq) > 1:
                    next_obs[:-1] = obs_seq[1:]
                next_obs_list.append(next_obs)

        return _TrialDataset(
            obs_list=obs_list,
            prev_act_list=prev_act_list,
            pos_list=pos_list,
            act_list=act_list,
            next_obs_list=next_obs_list,
        )

    def _compute_loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        obs = batch["obs"].to(self.device)
        prev_act = batch["prev_act"].to(self.device)
        lengths = batch["lengths"].to(self.device)
        act_target = batch["act"].to(self.device)

        pos_logits, act_logits, obs_logits = self.model(
            obs, prev_act, lengths=lengths, return_hidden=False
        )

        total_loss = torch.tensor(0.0, device=self.device)
        logs: dict[str, float] = {}

        if "pos" in batch:
            pos_target = batch["pos"].to(self.device)
            pos_loss = F.cross_entropy(
                pos_logits.reshape(-1, N_POS),
                pos_target.reshape(-1),
                ignore_index=IGNORE_INDEX,
            )
            total_loss = total_loss + self.pos_loss_weight * pos_loss
            logs["pos_loss"] = float(pos_loss.detach().cpu())

        act_loss = F.cross_entropy(
            act_logits.reshape(-1, self.n_action_classes),
            act_target.reshape(-1),
            ignore_index=IGNORE_INDEX,
        )
        total_loss = total_loss + self.act_loss_weight * act_loss
        logs["act_loss"] = float(act_loss.detach().cpu())

        if self.predict_next_obs and obs_logits is not None and "next_obs" in batch:
            next_obs_target = batch["next_obs"].to(self.device)
            obs_loss = F.cross_entropy(
                obs_logits.reshape(-1, N_OBS),
                next_obs_target.reshape(-1),
                ignore_index=IGNORE_INDEX,
            )
            total_loss = total_loss + self.obs_loss_weight * obs_loss
            logs["obs_loss"] = float(obs_loss.detach().cpu())

        logs["total_loss"] = float(total_loss.detach().cpu())
        return total_loss, logs

    def _run_training(
        self,
        dataset: _TrialDataset,
        n_iter: int,
        lr: Optional[float] = None,
        term_early: bool = False,
        verbose: bool = True,
    ) -> list[float]:
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=_collate_batch,
        )

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr if lr is None else lr,
            weight_decay=self.weight_decay,
        )

        progression: list[float] = []
        best = np.inf
        patience = 5
        bad_epochs = 0

        self.model.train()
        for epoch in range(n_iter):
            epoch_losses: list[float] = []

            for batch in loader:
                optimizer.zero_grad()
                loss, _ = self._compute_loss(batch)
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu()))

            mean_loss = float(np.mean(epoch_losses)) if epoch_losses else np.nan
            progression.append(mean_loss)

            if verbose:
                print(f"Epoch {epoch + 1}/{n_iter} | loss = {mean_loss:.6f}")

            if term_early:
                if mean_loss + 1e-8 < best:
                    best = mean_loss
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                if bad_epochs >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch + 1}.")
                    break

        return progression

    def _predict_position_trials(
        self,
        obs_arr: np.ndarray,
        act_arr: np.ndarray,
        trial_beg: Optional[np.ndarray] = None,
        trial_end: Optional[np.ndarray] = None,
        return_prob: bool = False,
    ) -> np.ndarray:
        dataset = self._build_dataset(
            obs_arr=obs_arr,
            act_arr=act_arr,
            pos_arr=None,
            trial_beg=trial_beg,
            trial_end=trial_end,
        )

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=_collate_batch,
        )

        outputs: list[np.ndarray] = []
        self.model.eval()

        with torch.no_grad():
            for batch in loader:
                obs = batch["obs"].to(self.device)
                prev_act = batch["prev_act"].to(self.device)
                lengths = batch["lengths"].to(self.device)

                pos_logits, _, _ = self.model(
                    obs, prev_act, lengths=lengths, return_hidden=False
                )

                if return_prob:
                    pos_out = F.softmax(pos_logits, dim=-1).cpu().numpy()
                else:
                    pos_out = pos_logits.argmax(dim=-1).cpu().numpy()

                lengths_np = lengths.cpu().numpy()
                for i, L in enumerate(lengths_np):
                    outputs.append(pos_out[i, :L])

        return np.concatenate(outputs, axis=0)

    # -------------------------
    # public API, mirroring CSCG
    # -------------------------
    def fit(self, n_iter: int = 100, term_early: bool = False) -> list[float]:
        """
        Fit the GRUCA3 model.
        If no explicit trial boundaries are provided, the whole sequence is treated
        as a single trial.
        """
        dataset = self._build_dataset(
            obs_arr=self.obs,
            act_arr=self.act,
            pos_arr=self.pos,
            trial_beg=None,
            trial_end=None,
        )
        return self._run_training(dataset, n_iter=n_iter, term_early=term_early)

    def fit_by_trial(
        self,
        trial_beg: np.ndarray,
        trial_end: np.ndarray,
        n_iter: int = 100,
        term_early: bool = False,
    ) -> list[float]:
        """
        Fit the GRUCA3 model treating each trial as a separate sequence.
        """
        dataset = self._build_dataset(
            obs_arr=self.obs,
            act_arr=self.act,
            pos_arr=self.pos,
            trial_beg=trial_beg,
            trial_end=trial_end,
        )
        return self._run_training(dataset, n_iter=n_iter, term_early=term_early)

    def predict(
        self,
        obs_test: np.ndarray,
        act_test: np.ndarray,
        trial_beg: Optional[np.ndarray] = None,
        trial_end: Optional[np.ndarray] = None,
        one_based: bool = True,
    ) -> np.ndarray:
        """
        Predict latent state / position from observation-action sequences.
        """
        pred = self._predict_position_trials(
            obs_arr=_ensure_int64(obs_test),
            act_arr=_ensure_int64(act_test),
            trial_beg=trial_beg,
            trial_end=trial_end,
            return_prob=False,
        ).astype(np.int64)

        return pred + 1 if one_based else pred

    def predict_with_plasticity(
        self,
        obs_test: np.ndarray,
        act_test: np.ndarray,
        pos_test: Optional[np.ndarray] = None,
        trial_beg: Optional[np.ndarray] = None,
        trial_end: Optional[np.ndarray] = None,
        n_iter: int = 10,
        lr: Optional[float] = None,
        term_early: bool = True,
        one_based: bool = True,
    ) -> np.ndarray:
        """
        Predict while allowing plasticity, implemented here as short online
        fine-tuning on the test sequence.

        Notes
        -----
        - If `pos_test` is provided, supervised fine-tuning is used.
        - If `pos_test` is None, the model still fine-tunes on action and optional
          next-observation prediction losses.
        """
        dataset = self._build_dataset(
            obs_arr=_ensure_int64(obs_test),
            act_arr=_ensure_int64(act_test),
            pos_arr=None if pos_test is None else _ensure_int64(pos_test),
            trial_beg=trial_beg,
            trial_end=trial_end,
        )
        self._run_training(
            dataset=dataset,
            n_iter=n_iter,
            lr=lr,
            term_early=term_early,
            verbose=False,
        )
        return self.predict(
            obs_test=obs_test,
            act_test=act_test,
            trial_beg=trial_beg,
            trial_end=trial_end,
            one_based=one_based,
        )

    def predict_prob(
        self,
        obs_test: np.ndarray,
        act_test: np.ndarray,
        trial_beg: Optional[np.ndarray] = None,
        trial_end: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Predict position probabilities for each timestep.

        Returns
        -------
        np.ndarray[np.float64]
            Shape (T, N_POS), concatenated over trials.
        """
        prob = self._predict_position_trials(
            obs_arr=_ensure_int64(obs_test),
            act_arr=_ensure_int64(act_test),
            trial_beg=trial_beg,
            trial_end=trial_end,
            return_prob=True,
        )
        return prob.astype(np.float64)

    def retrieve(
        self,
        obs_test: np.ndarray,
        act_test: np.ndarray,
        obs_perf: np.ndarray,
        act_perf: np.ndarray,
        trial_beg_test: Optional[np.ndarray] = None,
        trial_end_test: Optional[np.ndarray] = None,
        trial_beg_perf: Optional[np.ndarray] = None,
        trial_end_perf: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compare predicted latent states against a perfect/reference sequence.

        Returns
        -------
        np.ndarray[np.int64]
            1 for match and 0 for mismatch at each timestep.
        """
        state_test = self.predict(
            obs_test=obs_test,
            act_test=act_test,
            trial_beg=trial_beg_test,
            trial_end=trial_end_test,
            one_based=False,
        )
        state_perf = self.predict(
            obs_test=obs_perf,
            act_test=act_perf,
            trial_beg=trial_beg_perf,
            trial_end=trial_end_perf,
            one_based=False,
        )

        if state_test.shape[0] != state_perf.shape[0]:
            raise ValueError(
                "Predicted test and perfect sequences must have the same length "
                f"for retrieve(), got {state_test.shape[0]} vs {state_perf.shape[0]}."
            )

        return np.where(state_test == state_perf, 1, 0).astype(np.int64)

    def retrieve_trial_avg(
        self,
        trial_beg: np.ndarray,
        trial_end: np.ndarray,
        obs_test: np.ndarray,
        act_test: np.ndarray,
        pos_test: np.ndarray,
        obs_perf: np.ndarray,
        act_perf: np.ndarray,
        n_pos_bin: int = 144,
    ) -> np.ndarray:
        """
        Compute average retrieval for each spatial bin across trials.

        Notes
        -----
        `pos_test` should be spatial bin ids in 1..n_pos_bin.
        """
        obs_test = _ensure_int64(obs_test)
        act_test = _ensure_int64(act_test)
        pos_test = _ensure_int64(pos_test)
        trial_beg = _ensure_int64(trial_beg)
        trial_end = _ensure_int64(trial_end)

        _validate_trial_bounds(trial_beg, trial_end, len(obs_test))
        _check_same_length(obs_test, act_test, pos_test)

        avg_retrieval = np.zeros(n_pos_bin, dtype=np.float64)

        for beg, end in zip(trial_beg, trial_end):
            obs_test_trial = obs_test[beg:end]
            act_test_trial = act_test[beg:end]
            pos_test_trial = pos_test[beg:end]

            retrieval_trial = self.retrieve(
                obs_test=obs_test_trial,
                act_test=act_test_trial,
                obs_perf=obs_perf,
                act_perf=act_perf,
            )

            for pos in range(1, n_pos_bin + 1):
                mask = pos_test_trial == pos
                if np.any(mask):
                    avg_retrieval[pos - 1] += float(np.mean(retrieval_trial[mask]))

        avg_retrieval /= len(trial_beg)
        return avg_retrieval
    

if __name__ == "__main__":
    from rtrv_models.data.preprocess import preprocess_data
    
    res = preprocess_data(10212)