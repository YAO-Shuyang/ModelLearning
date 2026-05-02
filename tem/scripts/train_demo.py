from __future__ import annotations

import argparse
import torch
torch.set_num_threads(1)
from tqdm import trange

from rtrv_models.tem.envs.graph_world import GraphSpec, GraphWorld
from rtrv_models.tem.models.tem_lite import TEMConfig, TEMLite


def make_toy_spec() -> GraphSpec:
    # A 4x4 grid with heavily aliased observations: state type, not identity.
    width = 4
    n = width * width
    transitions = {}
    obs_by_state = []
    for s in range(n):
        r, c = divmod(s, width)
        degree = 0
        acts = {}
        if r > 0:
            acts[0] = s - width; degree += 1  # North
        if c < width - 1:
            acts[1] = s + 1; degree += 1      # East
        if r < width - 1:
            acts[2] = s + width; degree += 1  # South
        if c > 0:
            acts[3] = s - 1; degree += 1      # West
        transitions[s] = acts
        obs_by_state.append(1 if degree == 2 else 4 if degree >= 3 else 0)
    obs_by_state[-1] = 5  # goal-like node
    return GraphSpec(n_states=n, n_observations=6, n_actions=4, obs_by_state=obs_by_state, transitions=transitions)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    spec = make_toy_spec()
    world = GraphWorld(spec, seed=1)
    cfg = TEMConfig(n_observations=spec.n_observations, n_actions=spec.n_actions, n_states=spec.n_states)
    model = TEMLite(cfg).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    pbar = trange(args.steps)
    for step in pbar:
        obs, actions, states = world.sample_batch(args.batch_size, args.seq_len)
        obs, actions, states = obs.to(args.device), actions.to(args.device), states.to(args.device)
        out = model(obs, actions, states)
        opt.zero_grad()
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 20 == 0:
            with torch.no_grad():
                pred = out["state_logits"].argmax(-1)
                acc = (pred == states).float().mean().item()
            pbar.set_description(f"loss={out['loss'].item():.3f} state_acc={acc:.3f}")

    torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, "tem_lite_demo.pt")
    print("saved tem_lite_demo.pt")


if __name__ == "__main__":
    main()
