import gym


class GoalTerminateWrapper(gym.Wrapper):
    """Terminate when entering the goal node."""

    def __init__(self, env, goal_state: int) -> None:
        super().__init__(env)
        self.goal_state = int(goal_state)

    def step(self, action):
        obs, r, done, info = self.env.step(action)
        if int(obs) == self.goal_state:
            done = True
        return obs, r, done, info

class StepCostWrapper(gym.Wrapper):
    """Add constant step cost to each transition."""

    def __init__(self, env, step_cost: float) -> None:
        super().__init__(env)
        self.step_cost = float(step_cost)

    def step(self, action):
        obs, r, done, info = self.env.step(action)
        return obs, float(r) + self.step_cost, done, info
    
class StartReturnPenaltyWrapper(gym.Wrapper):
    """Add a punishment when the agent returns to the start state.

    Parameters
    ----------
    env
        Gym-like environment.
    start_state
        Start state index used in env.reset(agent_pos=...).
    penalty
        Negative reward added when the agent returns to start after
        having left it at least once in the current episode.

    Notes
    -----
    This wrapper does not punish being at the start initially. It only
    punishes transitions that land in `start_state` after the agent has
    visited any other state in the episode.

    """
    def __init__(
        self,
        env,
        start_state: int,
        penalty: float = -0.1,
    ) -> None:
        super().__init__(env)
        self.start_state = int(start_state)
        self.penalty = float(penalty)
        self._left_start = False

    def reset(self, **kwargs):
        """Reset and clear episode memory."""
        self._left_start = False
        return self.env.reset(**kwargs)

    def step(self, action):
        """Step and apply penalty if returning to start."""
        obs, r, done, info = self.env.step(action)
        s2 = int(obs)

        if s2 != self.start_state:
            self._left_start = True

        if self._left_start and s2 == self.start_state:
            r = float(r) + self.penalty

        return obs, float(r), bool(done), info