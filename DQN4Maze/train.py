from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rtrv_models.DQN4Maze.log import TrainMonitor


@dataclass
class Transition:
    """A single transition for replay."""
    s: int
    a: int
    r: float
    s2: int
    done: bool


@dataclass
class TransitionBatch:
    """A minibatch of transitions."""
    s: np.ndarray
    a: np.ndarray
    r: np.ndarray
    s2: np.ndarray
    done: np.ndarray


class ReplayBuffer:
    """A simple circular replay buffer.

    Parameters
    ----------
    capacity
        Maximum number of transitions stored.

    """
    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self.data: List[Transition] = []
        self.idx = 0

    def __len__(self) -> int:
        return len(self.data)

    def push(self, tr: Transition) -> None:
        """Insert a transition."""
        if len(self.data) < self.capacity:
            self.data.append(tr)
        else:
            self.data[self.idx] = tr
        self.idx = (self.idx + 1) % self.capacity

    def sample(self, batch_size: int) -> TransitionBatch:
        """Sample a minibatch uniformly."""
        n = len(self.data)
        if batch_size > n:
            raise ValueError(f"batch_size={batch_size} > buffer={n}")
        idx = np.random.choice(n, size=batch_size, replace=False)
        s = np.array([self.data[i].s for i in idx], dtype=np.int64)
        a = np.array([self.data[i].a for i in idx], dtype=np.int64)
        r = np.array([self.data[i].r for i in idx], dtype=np.float32)
        s2 = np.array([self.data[i].s2 for i in idx], dtype=np.int64)
        d = np.array([self.data[i].done for i in idx], dtype=np.float32)
        return TransitionBatch(s=s, a=a, r=r, s2=s2, done=d)


class QNet(nn.Module):
    """Embedding-based Q-network for discrete state indices.

    Parameters
    ----------
    n_states
        Number of discrete states.
    n_actions
        Number of actions.
    emb_dim
        Embedding dimension.
    hidden_dim
        Hidden layer dimension.

    """
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        emb_dim: int = 64,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.emb = nn.Embedding(n_states, emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """Compute Q(s, ·) for a batch of states."""
        z = self.emb(s)
        return self.mlp(z)


class DQNAgent:
    """Double DQN agent for Neuro-Nav graph tasks.

    Parameters
    ----------
    n_states
        Number of discrete environment states.
    n_actions
        Number of actions.
    lr
        Learning rate.
    gamma
        Discount factor.
    target_tau
        Polyak averaging factor. If 1.0, do hard copy.
    device
        Torch device string.

    Notes
    -----
    This implements Double DQN targets:
    a* = argmax_a Q_online(s', a)
    y  = r + (1-done)*gamma*Q_target(s', a*)

    """
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        target_tau: float = 1.0,
        device: str = "cpu",
    ) -> None:
        self.n_states = int(n_states)
        self.n_actions = int(n_actions)
        self.gamma = float(gamma)
        self.tau = float(target_tau)
        self.device = torch.device(device)

        self.q = QNet(n_states, n_actions).to(self.device)
        self.q_tgt = QNet(n_states, n_actions).to(self.device)
        self.q_tgt.load_state_dict(self.q.state_dict())
        self.q_tgt.eval()

        self.opt = optim.Adam(self.q.parameters(), lr=float(lr))

    @torch.no_grad()
    def act(self, s: int, eps: float, env=None) -> int:
        """Epsilon-greedy with illegal-action masking + random tie-break.

        Parameters
        ----------
        s
            Current state index.
        eps
            Exploration probability.
        env
            Optional env exposing env.edges for legality masking. We treat
            an action as illegal if env.edges[s][a] == s (self-loop).

        Returns
        -------
        a
            Selected action index.

        """
        s = int(s)

        st = torch.tensor([s], dtype=torch.long, device=self.device)
        q = self.q(st)[0].detach().cpu().numpy()

        if env is not None:
            q = q.copy()
            for a in range(self.n_actions):
                if int(env.edges[s][a]) == s:
                    q[a] = -np.inf

        # If all actions are illegal (should be rare), fall back.
        if not np.isfinite(np.max(q)):
            return int(np.random.randint(self.n_actions))

        # Exploration: random legal action.
        if np.random.rand() < float(eps):
            legal = np.flatnonzero(np.isfinite(q))
            return int(np.random.choice(legal))

        # Exploitation: random tie-break among best legal actions.
        q = q + np.random.randn(q.shape[0]).astype(np.float32) * 1e-6
        m = np.max(q)
        best = np.flatnonzero(np.isclose(q, m))
        return int(np.random.choice(best))

    def update(self, batch: TransitionBatch) -> float:
        """One Double DQN update step."""
        s = torch.tensor(batch.s, dtype=torch.long, device=self.device)
        a = torch.tensor(batch.a, dtype=torch.long, device=self.device)
        r = torch.tensor(batch.r, dtype=torch.float32, device=self.device)
        s2 = torch.tensor(batch.s2, dtype=torch.long, device=self.device)
        d = torch.tensor(batch.done, dtype=torch.float32, device=self.device)

        q_sa = self.q(s).gather(1, a[:, None]).squeeze(1)

        with torch.no_grad():
            # Double DQN: action selection by online network.
            a2 = torch.argmax(self.q(s2), dim=1)
            q2 = self.q_tgt(s2).gather(1, a2[:, None]).squeeze(1)
            y = r + (1.0 - d) * self.gamma * q2

        loss = nn.functional.smooth_l1_loss(q_sa, y)

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), max_norm=5.0)
        self.opt.step()

        self._update_target()
        return float(loss.item())

    def _update_target(self) -> None:
        """Update target network (hard or Polyak)."""
        if self.tau >= 1.0:
            self.q_tgt.load_state_dict(self.q.state_dict())
            return
        with torch.no_grad():
            for p, pt in zip(self.q.parameters(), self.q_tgt.parameters()):
                pt.data.mul_(1.0 - self.tau)
                pt.data.add_(self.tau * p.data)


def train_dqn(
    env,
    start_state: int,
    n_steps: int = 200_000,
    buffer_size: int = 200_000,
    warmup: int = 2_000,
    batch_size: int = 128,
    train_every: int = 4,
    target_tau: float = 1.0,
    gamma: float = 0.99,
    lr: float = 1e-3,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    eps_decay_steps: int = 100_000,
    max_ep_len: int = 2_000,
    device: str = "cpu",
    log_every: int = 2000,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    """Train Double DQN on a Neuro-Nav-like env with discrete states."""
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))

    mon = TrainMonitor(
        log_every=log_every,
        ret_window=50,
        loss_window=200,
    )

    n_states = int(env.state_size)
    n_actions = int(env.action_space.n)
    agent = DQNAgent(
        n_states=n_states,
        n_actions=n_actions,
        lr=lr,
        gamma=gamma,
        target_tau=target_tau,
        device=device,
    )
    buf = ReplayBuffer(buffer_size)

    ep_returns: List[float] = []
    losses: List[float] = []

    s = int(env.reset(agent_pos=int(start_state)))
    ep_ret = 0.0
    ep_len = 0

    records: List[List[tuple]] = []
    _rcds: List[tuple] = []

    for t in range(int(n_steps)):
        frac = min(1.0, t / max(1, int(eps_decay_steps)))
        eps = float(eps_start + frac * (eps_end - eps_start))

        a = agent.act(s, eps=eps, env=env)
        s2, r, done, info = env.step(int(a))
        s2 = int(s2)

        _rcds.append((int(s), int(s2), int(a), float(r)))
        mon.add_transition(int(s), int(a), int(s2))

        ep_ret += float(r)
        ep_len += 1

        buf.push(
            Transition(
                s=int(s),
                a=int(a),
                r=float(r),
                s2=int(s2),
                done=bool(done),
            )
        )

        s = s2

        if bool(done) or ep_len >= int(max_ep_len):
            ep_returns.append(float(ep_ret))
            mon.add_episode(float(ep_ret), int(ep_len))
            records.append(_rcds)
            _rcds = []
            s = int(env.reset(agent_pos=int(start_state)))
            ep_ret = 0.0
            ep_len = 0

        if len(buf) >= int(batch_size) and t >= int(warmup):
            if (t % int(train_every)) == 0:
                batch = buf.sample(int(batch_size))
                loss = agent.update(batch)
                losses.append(float(loss))
                mon.add_loss(float(loss))

        mon.maybe_log(
            step=t + 1,
            n_steps=int(n_steps),
            eps=float(eps),
            buf_len=len(buf),
            lr=float(lr),
        )

    logs = {
        "episode_return": np.asarray(ep_returns, dtype=np.float32),
        "loss": np.asarray(losses, dtype=np.float32),
        "records": records,
    }
    return logs, agent, records