import numpy as np


def rollout_greedy_alloc(
    env,
    agent,
    start_state: int,
    goal_state: int = 143,
    max_steps: int = 20000,
) -> dict:
    """Greedy rollout for allocentric env (obs is node index).

    Parameters
    ----------
    env
        Gym-like env returning int obs in [0, 143].
    agent
        Agent with act(s, eps).
    start_state
        Start state index (0-based).
    goal_state
        Goal state index (0-based). For node 144, this is 143.
    max_steps
        Max steps.

    Returns
    -------
    out
        Dict with states, actions, rewards, done, reached_goal.

    """
    s = int(env.reset(agent_pos=int(start_state)))

    states = [s]
    actions = []
    rewards = []

    reached_goal = False
    done = False

    for _ in range(int(max_steps)):
        a = int(agent.act(s, eps=0.0))
        s2, r, done, info = env.step(a)
        s2 = int(s2)

        actions.append(a)
        rewards.append(float(r))
        states.append(s2)

        s = s2
        if s == int(goal_state):
            reached_goal = True
        if done:
            break

    return {
        "states": np.asarray(states, dtype=np.int64),
        "actions": np.asarray(actions, dtype=np.int64),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "done": np.asarray(done, dtype=bool),
        "reached_goal": np.asarray(reached_goal, dtype=bool),
    }


def eval_steps_to_goal_alloc(
    env,
    agent,
    start_state: int,
    n_trials: int = 50,
    goal_state: int = 143,
    max_steps: int = 20000,
) -> np.ndarray:
    """Steps-to-goal over multiple greedy rollouts (allocentric)."""
    steps = np.empty((int(n_trials),), dtype=np.int64)
    for i in range(int(n_trials)):
        out = rollout_greedy_alloc(
            env=env,
            agent=agent,
            start_state=start_state,
            goal_state=goal_state,
            max_steps=max_steps,
        )
        if bool(out["reached_goal"]):
            steps[i] = int(out["actions"].size)
        else:
            steps[i] = int(max_steps)
    return steps