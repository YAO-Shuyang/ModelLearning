from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import gym


DIRS = ("N", "E", "S", "W")


def node_to_rc(node: int, width: int = 12) -> Tuple[int, int]:
    """Convert 1-based node id to (row, col) 0-based (your convention).

    Parameters
    ----------
    node
        Node id in {1, ..., width * width}.
    width
        Grid width.

    Returns
    -------
    r, c
        Row and column indices (0-based).

    Notes
    -----
    This follows your indexing:
    r = (node-1) % width
    c = (node-1) // width

    """
    node0 = int(node) - 1
    r = node0 % int(width)
    c = node0 // int(width)
    return int(r), int(c)


def infer_cardinal_dir(u: int, v: int, width: int = 12) -> str:
    """Infer cardinal direction of move u -> v using your node_to_rc.

    Parameters
    ----------
    u, v
        1-based node ids.
    width
        Grid width.

    Returns
    -------
    d
        One of {"N", "E", "S", "W"}.

    Raises
    ------
    ValueError
        If (u, v) is not a 4-neighbor move under your 12x12 convention.

    """
    ur, uc = node_to_rc(u, width)
    vr, vc = node_to_rc(v, width)
    dr, dc = vr - ur, vc - uc

    if (dr, dc) == (-1, 0):
        return "N"
    if (dr, dc) == (0, 1):
        return "E"
    if (dr, dc) == (1, 0):
        return "S"
    if (dr, dc) == (0, -1):
        return "W"

    raise ValueError(
        f"Edge {u}->{v} is not 4-neighbor under your indexing. "
        f"(dr,dc)=({dr},{dc})."
    )


def build_cardinal_neighbors(
    maze_graph: Dict[int, List[int]],
    width: int = 12,
) -> Dict[int, Dict[str, Optional[int]]]:
    """Build per-node cardinal neighbor map.

    Parameters
    ----------
    maze_graph
        Adjacency dict.
    width
        Grid width.

    Returns
    -------
    cardinal
        cardinal[u][d] = neighbor node or None.

    """
    cardinal = {u: {d: None for d in DIRS} for u in maze_graph}
    for u, nbrs in maze_graph.items():
        for v in nbrs:
            d = infer_cardinal_dir(u, v, width)
            cardinal[u][d] = v

    for u, nbrs in maze_graph.items():
        for v in nbrs:
            if u not in maze_graph.get(v, []):
                raise ValueError(f"Graph not symmetric: {u}->{v} only.")
    return cardinal


def build_allocentric_edges(
    maze_graph: Dict[int, List[int]],
    width: int = 12,
) -> List[List[int]]:
    """Build Neuro-Nav edges for allocentric actions N/E/S/W.

    State is node index in [0, n_nodes). Action is one of:
    0=N, 1=E, 2=S, 3=W.

    Parameters
    ----------
    maze_graph
        Adjacency dict with nodes 1..n_nodes.
    width
        Grid width.

    Returns
    -------
    edges
        edges[s][a] = s_next, using 0-based state indices.

    """
    n_nodes = len(maze_graph)
    cardinal = build_cardinal_neighbors(maze_graph, width)
    edges: List[List[int]] = [[0, 0, 0, 0] for _ in range(n_nodes)]

    for node in range(1, n_nodes + 1):
        s = node - 1
        for a, d in enumerate(DIRS):
            nxt = cardinal[node][d]
            if nxt is None:
                edges[s][a] = s
            else:
                edges[s][a] = nxt - 1
    return edges


@dataclass(frozen=True)
class MazeRewards:
    """Reward specification."""
    goal_reward: float = 1.0
    dead_end_punish: float = -0.05
    step_cost: float = 0.0


def find_leaf_nodes(
    maze_graph: Dict[int, List[int]],
    start_node: int,
    goal_node: int,
) -> List[int]:
    """Return leaf nodes (degree 1) excluding start and goal."""
    leaves = []
    for n, nbrs in maze_graph.items():
        if n in (start_node, goal_node):
            continue
        if len(nbrs) == 1:
            leaves.append(n)
    return leaves


def make_reward_table_allocentric(
    maze_graph: Dict[int, List[int]],
    start_node: int = 1,
    goal_node: int = 144,
    rewards: MazeRewards = MazeRewards(),
) -> Dict[int, float]:
    """Create reward dict keyed by 0-based state index."""
    reward_table: Dict[int, float] = {}
    reward_table[goal_node - 1] = float(rewards.goal_reward)

    leaves = find_leaf_nodes(maze_graph, start_node, goal_node)
    for leaf in leaves:
        reward_table[leaf - 1] = float(rewards.dead_end_punish)
    return reward_table
