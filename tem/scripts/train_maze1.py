from __future__ import annotations

import argparse
import torch
from tqdm import trange

from rtrv_models.tem.envs.graph_world import GraphWorld
from rtrv_models.tem.envs.maze1_adapter import load_maze1_allocentric_spec
from rtrv_models.tem.models.tem_lite import TEMConfig, TEMLite


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--start-node", type=int, default=1)
    parser.add_argument("--goal-node", type=int, default=144)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=str, default="tem_maze1.pt")
    args = parser.parse_args()

    spec = load_maze1_allocentric_spec(start_node=args.start_node, goal_node=args.goal_node)
    world = GraphWorld(spec, seed=1)
    cfg = TEMConfig(
        n_observations=spec.n_observations,
        n_actions=spec.n_actions,
        n_states=spec.n_states,
    )
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

    torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
