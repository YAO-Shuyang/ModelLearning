from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


DIRS = ("N", "E", "S", "W")


def node_to_rc(node: int, width: int = 12) -> Tuple[int, int]:
    """Convert a 1-based node id to 0-based (row, col).

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

    """
    node0 = node - 1
    return node0 // width, node0 % width


def infer_cardinal_dir(u: int, v: int, width: int = 12) -> str:
    """Infer the cardinal direction of a 4-neighbor move u -> v.

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
        If (u, v) is not a 4-neighbor move on the width x width grid.

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
        f"Edge {u}->{v} is not 4-neighbor on {width}x{width}. "
        f"(dr,dc)=({dr},{dc})."
    )


def build_cardinal_neighbors(
    maze_graph: Dict[int, List[int]],
    width: int = 12,
) -> Dict[int, Dict[str, Optional[int]]]:
    """Convert adjacency dict into a per-node N/E/S/W neighbor table.

    Parameters
    ----------
    maze_graph
        Keys are node ids (1-based). Values are lists of neighbor node ids.
    width
        Grid width.

    Returns
    -------
    cardinal
        cardinal[u][d] is the neighbor node in direction d, or None if
        blocked by a wall.

    Notes
    -----
    This function also sanity-checks that `maze_graph` is symmetric
    (undirected doors): u in neighbors[v] whenever v in neighbors[u].

    """
    cardinal: Dict[int, Dict[str, Optional[int]]] = {
        u: {d: None for d in DIRS} for u in maze_graph
    }

    for u, nbrs in maze_graph.items():
        for v in nbrs:
            d = infer_cardinal_dir(u, v, width)
            prev = cardinal[u][d]
            if prev is not None and prev != v:
                raise ValueError(
                    f"Node {u} has two neighbors in direction {d}: "
                    f"{prev} and {v}."
                )
            cardinal[u][d] = v

    for u, nbrs in maze_graph.items():
        for v in nbrs:
            if u not in maze_graph.get(v, []):
                raise ValueError(
                    f"Graph not symmetric: {u}->{v} exists but "
                    f"{v}->{u} does not."
                )

    return cardinal


def sid(node: int, heading: int, n_nodes: int) -> int:
    """Map (node, heading) to an oriented state id.

    Parameters
    ----------
    node
        1-based node id in {1, ..., n_nodes}.
    heading
        Integer in {0, 1, 2, 3} meaning N/E/S/W.
    n_nodes
        Number of physical nodes (144 for a 12x12 grid).

    Returns
    -------
    s
        Oriented state id in {0, ..., 4*n_nodes - 1}.

    """
    return (node - 1) * 4 + heading


def oriented_edges_ego_move(
    maze_graph: Dict[int, List[int]],
    width: int = 12,
) -> List[List[int]]:
    """Build oriented-state transitions for egocentric actions.

    The state is (node, heading). The action set is:

    - 0: Forward
    - 1: Left (rotate left, then move)
    - 2: Right (rotate right, then move)
    - 3: U-turn (rotate 180, then move)

    If the move in the resulting heading is blocked by a wall, the
    transition is a self-loop.

    Parameters
    ----------
    maze_graph
        Adjacency dict of the maze.
    width
        Grid width.

    Returns
    -------
    edges
        List of length 4*n_nodes. edges[s] is a list of length 4 with
        next-state ids.

    """
    n_nodes = len(maze_graph)
    cardinal = build_cardinal_neighbors(maze_graph, width)

    rot = np.array([0, -1, +1, +2], dtype=int)
    edges: List[List[int]] = [[0, 0, 0, 0] for _ in range(4 * n_nodes)]

    for node in range(1, n_nodes + 1):
        for h in range(4):
            s = sid(node, h, n_nodes)
            for a in range(4):
                h2 = int((h + rot[a]) % 4)
                d2 = DIRS[h2]
                nxt = cardinal[node][d2]
                if nxt is None:
                    edges[s][a] = s
                else:
                    edges[s][a] = sid(nxt, h2, n_nodes)

    return edges


def find_leaf_nodes(
    maze_graph: Dict[int, List[int]],
    start_node: int,
    goal_node: int,
) -> List[int]:
    """Identify side-branch ends as leaves (degree 1) excluding endpoints.

    Parameters
    ----------
    maze_graph
        Adjacency dict of the maze.
    start_node
        Start node id.
    goal_node
        Goal node id.

    Returns
    -------
    leaves
        Nodes with exactly one neighbor, excluding start and goal.

    Notes
    -----
    This matches the common definition of a cul-de-sac in a graph maze.
    If you have a different definition (e.g., "dead ends in side branches
    relative to a backbone path"), we can adjust this.

    """
    leaves = []
    for n, nbrs in maze_graph.items():
        if n in (start_node, goal_node):
            continue
        if len(nbrs) == 1:
            leaves.append(n)
    return leaves


@dataclass(frozen=True)
class MazeRewards:
    """Reward specification for the maze."""
    goal_reward: float = 1.0
    dead_end_punish: float = -0.2
    step_cost: float = 0.0


def make_reward_table(
    maze_graph: Dict[int, List[int]],
    start_node: int = 1,
    goal_node: int = 144,
    rewards: MazeRewards = MazeRewards(),
) -> Dict[int, float]:
    """Create a Neuro-Nav-style reward dict keyed by oriented state id.

    Parameters
    ----------
    maze_graph
        Adjacency dict of the maze.
    start_node
        Start node id.
    goal_node
        Goal node id.
    rewards
        Reward parameters.

    Returns
    -------
    reward_table
        Maps oriented state ids to scalar reward. Any state not in the
        dict has reward 0 (GraphEnv behavior).

    """
    n_nodes = len(maze_graph)
    reward_table: Dict[int, float] = {}

    for h in range(4):
        reward_table[sid(goal_node, h, n_nodes)] = rewards.goal_reward

    leaves = find_leaf_nodes(maze_graph, start_node, goal_node)
    for leaf in leaves:
        for h in range(4):
            reward_table[sid(leaf, h, n_nodes)] = rewards.dead_end_punish

    return reward_table


def make_start_state(
    maze_graph: Dict[int, List[int]],
    start_node: int = 1,
    start_heading: int = 1,
) -> int:
    """Return the oriented start state id.

    Parameters
    ----------
    maze_graph
        Adjacency dict of the maze.
    start_node
        Start node id.
    start_heading
        Heading in {0,1,2,3} meaning N/E/S/W. Default is East (1).

    Returns
    -------
    s0
        Oriented start state id.

    """
    n_nodes = len(maze_graph)
    return sid(start_node, start_heading, n_nodes)


def nodes_to_ego_actions(
    nodes: Sequence[int],
    width: int = 12,
) -> np.ndarray:
    """Convert a node trajectory into egocentric actions by turn rule.

    The egocentric action at time t is defined by the relative change
    from the previous movement direction to the next movement direction.
    Example: East then South -> turning right.

    Action coding:
    - 0: Forward (same direction)
    - 1: Left
    - 2: Right
    - 3: U-turn

    Parameters
    ----------
    nodes
        Sequence of visited nodes [n0, n1, ..., nT] (1-based).
    width
        Grid width.

    Returns
    -------
    actions
        Integer array of length T-2 with values in {0,1,2,3}. The first
        egocentric action corresponds to the turn from move (n0->n1) to
        move (n1->n2).

    """
    if len(nodes) < 3:
        return np.zeros((0,), dtype=np.int64)

    dir_to_h = {"N": 0, "E": 1, "S": 2, "W": 3}

    move_h: List[int] = []
    for t in range(len(nodes) - 1):
        d = infer_cardinal_dir(nodes[t], nodes[t + 1], width)
        move_h.append(dir_to_h[d])

    acts = np.empty((len(move_h) - 1,), dtype=np.int64)
    for t in range(len(move_h) - 1):
        prev_h = move_h[t]
        next_h = move_h[t + 1]
        delta = (next_h - prev_h) % 4
        if delta == 0:
            acts[t] = 0
        elif delta == 3:
            acts[t] = 1
        elif delta == 1:
            acts[t] = 2
        else:
            acts[t] = 3
    return acts


def build_neuronav_objects_edges_start(
    maze_graph: Dict[int, List[int]],
    start_node: int = 1,
    goal_node: int = 144,
    start_heading: int = 1,
    rewards: MazeRewards = MazeRewards(),
    width: int = 12,
) -> Tuple[Dict, List[List[int]], int]:
    """Build (objects, edges, start_state) for Neuro-Nav GraphEnv.

    Parameters
    ----------
    maze_graph
        Adjacency dict of the maze.
    start_node
        Start node id.
    goal_node
        Goal node id.
    start_heading
        Start heading in {0,1,2,3} meaning N/E/S/W.
    rewards
        Reward parameters.
    width
        Grid width.

    Returns
    -------
    objects
        Dict containing "rewards" table.
    edges
        Oriented transition table (4 actions).
    start_state
        Oriented start state id.

    """
    edges = oriented_edges_ego_move(maze_graph, width)
    reward_table = make_reward_table(
        maze_graph,
        start_node=start_node,
        goal_node=goal_node,
        rewards=rewards,
    )
    objects = {"rewards": reward_table}
    start_state = make_start_state(
        maze_graph,
        start_node=start_node,
        start_heading=start_heading,
    )
    return objects, edges, start_state