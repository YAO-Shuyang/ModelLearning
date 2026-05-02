from __future__ import annotations

# Egocentric action convention matching the user's formulation:
# 0 Start/No-op, 1 Left, 2 Forward, 3 Right, 4 Backward, 5 Goal reached.
# Heading convention: 0 North, 1 East, 2 South, 3 West.

TURN = {
    0: 0,   # start/no-op
    1: -1,  # left
    2: 0,   # forward
    3: 1,   # right
    4: 2,   # backward
    5: 0,   # terminal/goal
}


def update_heading(heading: int, ego_action: int) -> int:
    return (int(heading) + TURN[int(ego_action)]) % 4


def ego_to_allocentric(heading: int, ego_action: int) -> int:
    """Return allocentric movement direction after applying an ego action.

    For turns, this returns the new heading. Whether the animal actually moves
    on a turn depends on your environment convention; many maze tasks treat
    turn and movement separately.
    """
    return update_heading(heading, ego_action)
