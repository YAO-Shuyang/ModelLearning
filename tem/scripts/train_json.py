from __future__ import annotations

import argparse
import torch
torch.set_num_threads(1)
from tqdm import trange

from rtrv_models.tem.envs.graph_world import GraphSpec, GraphWorld
from rtrv_models.tem.models.tem_lite import TEMConfig, TEMLite


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True, help="Path to graph JSON")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=str, default="tem_lite.pt")
    args = parser.parse_args()

    spec = GraphSpec.from_json(args.env)
    world = GraphWorld(spec, seed=1)
    cfg = TEMConfig(n_observations=spec.n_observations, n_actions=spec.n_actions, n_states=spec.n_states)
    model = TEMLite(cfg).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for step in trange(args.steps):
        obs, actions, states = world.sample_batch(args.batch_size, args.seq_len)
        obs, actions, states = obs.to(args.device), actions.to(args.device), states.to(args.device)
        out = model(obs, actions, states)
        opt.zero_grad()
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
