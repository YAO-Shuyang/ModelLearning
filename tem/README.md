# torch_tem_rebuilt

A clean, minimal PyTorch rebuild inspired by the Tolman-Eichenbaum Machine (TEM) repository.

This version is designed for hippocampal retrieval experiments with aliased observations and either allocentric or egocentric actions.
It is not a line-by-line copy of `jbakermans/torch_tem`; instead, it preserves the key computational idea:

1. learn a reusable transition/path-integration code `g_t` from actions,
2. bind that abstract code to sensory observations through a conjunctive hippocampal-like code `p_t`,
3. update a fast Hebbian memory within each episode/environment,
4. retrieve from memory to improve belief over latent state.

## Installation

```bash
pip install -e .
```

## Run a demo

```bash
python -m tem_torch_rebuilt.scripts.train_demo --steps 2000 --device cpu
```

## Adapt to your maze

Prepare a JSON file with:

```json
{
  "n_states": 144,
  "n_observations": 6,
  "n_actions": 4,
  "obs_by_state": [0, 2, 2, ...],
  "transitions": {"0": {"1": 1, "2": 12}, "1": {"1": 2, "3": 0}}
}
```

where `transitions[state][action] = next_state`. For egocentric actions, include orientation in the state or use the helper in `envs/ego.py`.
