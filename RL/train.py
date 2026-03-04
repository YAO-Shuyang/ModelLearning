from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from neuronav.envs.graph_env import GraphObservation

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym


@dataclass
class ResetOut:
    """Reset output container."""
    obs: Any
    info: Dict[str, Any]


class OldGymAPIWrapper(gym.Wrapper):
    """Wrap a Gymnasium env to expose old Gym-style API.

    This wrapper converts:
    - reset() -> obs  (drops info)
    - step()  -> (obs, reward, done, info)

    Parameters
    ----------
    env
        A Gymnasium environment.

    """
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)

    def reset(self, **kwargs) -> Any:
        """Reset and return obs only (old Gym API)."""
        obs, info = self.env.reset(**kwargs)
        return obs

    def step(self, action: int) -> Tuple[Any, float, bool, Dict[str, Any]]:
        """Step and return (obs, reward, done, info)."""
        obs, r, term, trunc, info = self.env.step(action)
        done = bool(term or trunc)
        return obs, float(r), done, info

@dataclass
class Transition:
    """A single transition for replay."""
    s: int
    a: int
    r: float
    s2: int
    done: bool


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


@dataclass
class TransitionBatch:
    """A minibatch of transitions."""
    s: np.ndarray
    a: np.ndarray
    r: np.ndarray
    s2: np.ndarray
    done: np.ndarray


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
        """Compute Q(s, ·) for a batch of states.

        Parameters
        ----------
        s
            Long tensor of shape (B,) containing state indices.

        Returns
        -------
        q
            Float tensor of shape (B, n_actions).

        """
        z = self.emb(s)
        return self.mlp(z)


class DQNAgent:
    """DQN agent for Neuro-Nav graph tasks.

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

        self.opt = optim.Adam(self.q.parameters(), lr=lr)

    @torch.no_grad()
    def act(self, s: int, eps: float) -> int:
        """Epsilon-greedy action selection."""
        if np.random.rand() < eps:
            return int(np.random.randint(self.n_actions))
        st = torch.tensor([s], dtype=torch.long, device=self.device)
        q = self.q(st)[0]
        return int(torch.argmax(q).item())

    def update(self, batch: TransitionBatch) -> float:
        """One DQN update step.

        Parameters
        ----------
        batch
            Minibatch of transitions.

        Returns
        -------
        loss_value
            Scalar loss as float.

        """
        s = torch.tensor(batch.s, dtype=torch.long, device=self.device)
        a = torch.tensor(batch.a, dtype=torch.long, device=self.device)
        r = torch.tensor(batch.r, dtype=torch.float32, device=self.device)
        s2 = torch.tensor(batch.s2, dtype=torch.long, device=self.device)
        d = torch.tensor(batch.done, dtype=torch.float32, device=self.device)

        q_sa = self.q(s).gather(1, a[:, None]).squeeze(1)

        with torch.no_grad():
            q2_max = self.q_tgt(s2).max(dim=1).values
            y = r + (1.0 - d) * self.gamma * q2_max

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
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    """Train DQN on a Neuro-Nav-like env with discrete state indices.

    Parameters
    ----------
    env
        A Gym-like env. Must support reset(agent_pos=...) and step(a).
        Observation must be an integer state index.
    start_state
        Oriented start state id.
    n_steps
        Total environment steps.
    buffer_size
        Replay buffer capacity.
    warmup
        Steps to fill buffer before training.
    batch_size
        Minibatch size.
    train_every
        Update frequency in environment steps.
    target_tau
        Target update: 1.0 for hard copy each update, <1 for Polyak.
    gamma
        Discount factor.
    lr
        Learning rate.
    eps_start, eps_end
        Epsilon schedule endpoints.
    eps_decay_steps
        Linear decay duration in steps.
    max_ep_len
        Episode truncation length.
    device
        Torch device.
    seed
        Random seed.

    Returns
    -------
    logs
        Dict containing arrays of episode returns and losses.

    """
    np.random.seed(seed)
    torch.manual_seed(seed)

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

    s = int(env.reset(agent_pos=start_state))
    ep_ret = 0.0
    ep_len = 0

    for t in range(n_steps):
        frac = min(1.0, t / max(1, eps_decay_steps))
        eps = eps_start + frac * (eps_end - eps_start)

        a = agent.act(s, eps=eps)
        s2, r, done, info = env.step(a)
        s2 = int(s2)

        ep_ret += float(r)
        ep_len += 1

        tr = Transition(s=s, a=a, r=float(r), s2=s2, done=bool(done))
        buf.push(tr)

        s = s2

        if done or ep_len >= max_ep_len:
            ep_returns.append(ep_ret)
            s = int(env.reset(agent_pos=start_state))
            ep_ret = 0.0
            ep_len = 0

        if t >= warmup and (t % train_every == 0):
            batch = buf.sample(batch_size)
            loss = agent.update(batch)
            losses.append(loss)

    logs = {
        "episode_return": np.asarray(ep_returns, dtype=np.float32),
        "loss": np.asarray(losses, dtype=np.float32),
    }
    return logs

if __name__ == "__main__":
    import gymnasium as gym
    from dsp_models.RL.graph import CustomGraphEnv, maze1_graph
    from dsp_models.RL.utils import build_neuronav_objects_edges_start, MazeRewards
    
    # 2) Then run the builder
    rewards = MazeRewards(goal_reward=1.0, dead_end_punish=-0.2)
    objects, edges, s0 = build_neuronav_objects_edges_start(
        maze1_graph,
        start_node=1,
        goal_node=144,
        start_heading=1,
        rewards=rewards,
        width=12,
    )
    # Instantiate Neuro-Nav env with integer observations.
    env0 = CustomGraphEnv(objects, edges, obs_type=GraphObservation.index)
    env = OldGymAPIWrapper(env0)
    logs = train_dqn(
        env=env,
        start_state=s0,
        n_steps=200_000,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    print("Episodes:", len(logs["episode_return"]))
    print("Last 10 returns:", logs["episode_return"][-10:])