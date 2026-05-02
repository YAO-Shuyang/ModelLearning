from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import json
import random
import numpy as np
import torch


@dataclass
class GraphSpec:
    n_states: int
    n_observations: int
    n_actions: int
    obs_by_state: List[int]
    transitions: Dict[int, Dict[int, int]]

    @staticmethod
    def from_json(path: str) -> "GraphSpec":
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return GraphSpec(
            n_states=int(raw["n_states"]),
            n_observations=int(raw["n_observations"]),
            n_actions=int(raw["n_actions"]),
            obs_by_state=[int(x) for x in raw["obs_by_state"]],
            transitions={int(s): {int(a): int(ns) for a, ns in acts.items()}
                         for s, acts in raw["transitions"].items()},
        )


class GraphWorld:
    """Sample action-observation-position sequences from a discrete graph.

    Observations may be aliased: multiple latent states can map to the same
    observation symbol. The model therefore must use action history and memory
    to infer position.
    """

    def __init__(self, spec: GraphSpec, seed: Optional[int] = None):
        self.spec = spec
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

    def sample_sequence(self, length: int, start_state: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if start_state is None:
            state = self.rng.randrange(self.spec.n_states)
        else:
            state = int(start_state)

        states, obs, actions = [], [], []
        for _ in range(length):
            states.append(state)
            obs.append(self.spec.obs_by_state[state])
            legal = list(self.spec.transitions.get(state, {}).keys())
            if not legal:
                action = self.rng.randrange(self.spec.n_actions)
                next_state = state
            else:
                action = self.rng.choice(legal)
                next_state = self.spec.transitions[state][action]
            actions.append(action)
            state = next_state

        return (
            torch.tensor(obs, dtype=torch.long),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(states, dtype=torch.long),
        )

    def sample_batch(self, batch_size: int, length: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seqs = [self.sample_sequence(length) for _ in range(batch_size)]
        obs, actions, states = zip(*seqs)
        return torch.stack(obs, 0), torch.stack(actions, 0), torch.stack(states, 0)
