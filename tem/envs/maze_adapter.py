from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from rtrv_models.data.preprocess import PreprocessedData, preprocess_data
from rtrv_models.tem.envs.graph_world import GraphSpec

WIDTH = 12
NORTH = 0
EAST = 1
SOUTH = 2
WEST = 3
GOAL_REACHED = 4

ActionMode = Literal["allocentric", "egocentric"]


def node_to_xy(node: int, width: int = WIDTH) -> tuple[int, int]:
    """Convert a 1-indexed maze node to 0-indexed (x, y)."""
    return (node - 1) % width, (node - 1) // width


def allocentric_action(node: int, next_node: int) -> int:
    """Return allocentric action using rtrv_models' native convention.

    Action coding follows rtrv_models.data.graph.check_relation:
    0 North, 1 East, 2 South, 3 West.
    """
    from rtrv_models.data.graph import check_relation

    return int(check_relation(node, next_node)[0])


def observation_type(
    node: int,
    neighbors: list[int],
    *,
    start_node: int = 1,
    goal_node: int = 144,
    width: int = WIDTH,
) -> int:
    """Map a node to an aliased observation class.

    Observation coding:
    0 start node
    1 branch-end node
    2 straight node
    3 turning node
    4 junction node
    5 goal node
    """
    if node == start_node:
        return 0
    if node == goal_node:
        return 5

    degree = len(neighbors)

    if degree <= 1:
        return 1
    if degree >= 3:
        return 4

    # degree == 2: straight if the two neighbors lie on the same row or column.
    (x0, y0) = node_to_xy(neighbors[0], width)
    (x1, y1) = node_to_xy(neighbors[1], width)

    if x0 == x1 or y0 == y1:
        return 2

    return 3


def graph_to_allocentric_spec(
    graph: Dict[int, List[int]],
    *,
    start_node: int = 1,
    goal_node: int = 144,
    width: int = WIDTH,
    include_goal_action: bool = True,
) -> GraphSpec:
    """Convert rtrv_models.data.graph.maze1_graph into a GraphSpec.

    The input graph is 1-indexed; GraphSpec is 0-indexed internally.
    """
    nodes = sorted(graph)

    if nodes != list(range(1, len(nodes) + 1)):
        raise ValueError("Expected graph nodes to be contiguous and 1-indexed.")

    transitions: dict[int, dict[int, int]] = {}
    obs_by_state: list[int] = []

    for node in nodes:
        neighbors = list(graph[node])

        obs_by_state.append(
            observation_type(
                node,
                neighbors,
                start_node=start_node,
                goal_node=goal_node,
                width=width,
            )
        )

        state = node - 1
        transitions[state] = {}

        for next_node in neighbors:
            action = allocentric_action(node, next_node)
            transitions[state][action] = next_node - 1

    n_actions = 5 if include_goal_action else 4

    if include_goal_action:
        transitions[goal_node - 1][GOAL_REACHED] = goal_node - 1

    return GraphSpec(
        n_states=len(nodes),
        n_observations=6,
        n_actions=n_actions,
        obs_by_state=obs_by_state,
        transitions=transitions,
    )


def load_maze1_allocentric_spec(
    *,
    start_node: int = 1,
    goal_node: int = 144,
    width: int = WIDTH,
) -> GraphSpec:
    """Load maze1_graph from the parent rtrv_models repository.

    Run scripts from the repository root, or install the repository so that
    `rtrv_models.data.graph` is importable.
    """
    from rtrv_models.data.graph import maze1_graph

    return graph_to_allocentric_spec(
        maze1_graph,
        start_node=start_node,
        goal_node=goal_node,
        width=width,
    )


@dataclass
class MazeSequenceBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    mask: torch.Tensor
    states: torch.Tensor | None = None


class RealMazeSequenceDataset(Dataset):
    """Dataset wrapper for empirical action-observation sequences.

    Each item is one lap/trial segment. If `seq_len` is provided, the
    segment is randomly cropped or padded to fixed length.

    Returned item:
        obs:     LongTensor, shape (L,)
        actions: LongTensor, shape (L,)
        mask:    BoolTensor, shape (L,), True for real samples, False for pad
        states:  optional LongTensor, shape (L,)
    """

    def __init__(
        self,
        mouse: int,
        action_mode: ActionMode = "allocentric",
        maze_id: int | None = 1,
        seq_len: int | None = None,
        include_state: bool = False,
    ) -> None:
        self.mouse = mouse
        self.action_mode = action_mode
        self.maze_id = maze_id
        self.seq_len = seq_len
        self.include_state = include_state

        self.res: PreprocessedData = preprocess_data(mouse)

        self.obs = np.asarray(self.res.obs_train, dtype=np.int64)

        if action_mode == "allocentric":
            self.actions = np.asarray(self.res.all_act_train, dtype=np.int64)
        elif action_mode == "egocentric":
            self.actions = np.asarray(self.res.ego_act_train, dtype=np.int64)
        else:
            raise ValueError(f"Unknown action_mode: {action_mode}")

        self.beg, self.end = self.res.get_lap_dur_train()
        self.beg = np.asarray(self.beg, dtype=np.int64)
        self.end = np.asarray(self.end, dtype=np.int64)

        if maze_id is not None:
            maze_train = np.asarray(self.res.maze_train)
            keep = maze_train[self.beg] == maze_id
            self.beg = self.beg[keep]
            self.end = self.end[keep]

        self.states = None

        if include_state:
            for candidate in ["nodes_train", "pos_train", "state_train", "loc_train"]:
                if hasattr(self.res, candidate):
                    self.states = np.asarray(
                        getattr(self.res, candidate),
                        dtype=np.int64,
                    )
                    break

            if self.states is None:
                raise AttributeError(
                    "include_state=True, but no node/state field was found in "
                    "PreprocessedData. Please expose true node labels as, e.g., "
                    "`res.nodes_train`, or pass include_state=False."
                )

    def __len__(self) -> int:
        return len(self.beg)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        b = int(self.beg[idx])
        e = int(self.end[idx])

        obs = self.obs[b:e]
        actions = self.actions[b:e]

        start, stop, pad_len = self._sample_window(len(obs))

        obs_tensor, mask = self._crop_pad(obs, start, stop, pad_len)
        actions_tensor, _ = self._crop_pad(actions, start, stop, pad_len)

        item: dict[str, torch.Tensor] = {
            "obs": obs_tensor,
            "actions": actions_tensor,
            "mask": mask,
        }

        if self.states is not None:
            states = self.states[b:e]
            states_tensor, _ = self._crop_pad(states, start, stop, pad_len)
            item["states"] = states_tensor

        return item

    def _sample_window(self, length: int) -> tuple[int, int, int]:
        """Return crop start, crop stop, and pad length."""
        if self.seq_len is None:
            return 0, length, 0

        if length >= self.seq_len:
            start = np.random.randint(0, length - self.seq_len + 1)
            stop = start + self.seq_len
            pad_len = 0
        else:
            start = 0
            stop = length
            pad_len = self.seq_len - length

        return start, stop, pad_len

    def _crop_pad(
        self,
        arr: np.ndarray,
        start: int,
        stop: int,
        pad_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Crop/pad an array and return tensor plus validity mask."""
        arr = np.asarray(arr, dtype=np.int64)
        arr = arr[start:stop]

        valid_len = len(arr)
        mask = np.ones(valid_len, dtype=bool)

        if pad_len > 0:
            arr = np.pad(
                arr,
                (0, pad_len),
                mode="constant",
                constant_values=0,
            )
            mask = np.pad(
                mask,
                (0, pad_len),
                mode="constant",
                constant_values=False,
            )

        return (
            torch.as_tensor(arr, dtype=torch.long),
            torch.as_tensor(mask, dtype=torch.bool),
        )