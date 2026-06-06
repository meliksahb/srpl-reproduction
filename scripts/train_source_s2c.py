"""Train a source S2C model on PointButton1 and save it (for the Fig 6 transfer).

This is the SOURCE phase of the transferability experiment. We collect
trajectories on SafetyPointButton1-v0, label each state with its steps-to-cost
delta (the paper's labeling), and train an S2C model by the NLL/cross-entropy
loss of eq. (3). The trained S2C is saved to a checkpoint that the TARGET phase
loads (frozen) and applies to SafetyPointGoal1-v0.

Why standalone (decoupled from OmniSafe): the transfer only needs the S2C's
*weights*, and training the S2C directly here -- rather than extracting it from
an OmniSafe checkpoint -- is far more robust and fully reproducible. We use the
same labeling (`srpl.labeling`) and model (`srpl.s2c_model`) as the main runs, so
the representation is identical in form to the one learned online.

Data-collection policy: by default a RANDOM policy. The paper trains the S2C on
a diverse set of policies precisely so the representation is state-centric and
policy-agnostic; a random policy gives broad coverage of PointButton1's hazard
geometry and reliably triggers cost events for labeling. This is a documented
simplification of the paper's "S2C trained jointly with the source policy"; it
is sufficient to test whether a frozen source representation transfers.

Run:
    conda activate srpl && unset PYTHONPATH && cd ~/srpl-reproduction
    PYTHONPATH=. python scripts/train_source_s2c.py \
        --episodes 400 --epochs 30 --out experiments/transfer/s2c_button1.pt
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import safety_gymnasium
import torch

from srpl.labeling import (
    DEFAULT_SAFETY_HORIZON,
    DEFAULT_BIN_SIZE,
    label_trajectory,
)
from srpl.s2c_model import S2CModel


def collect_labeled_data(
    env_id: str,
    episodes: int,
    safety_horizon: int,
    bin_size: int,
    seed: int,
):
    """Roll out a random policy and return (obs, bin_label) arrays.

    Labeling mirrors the online wrapper: cost[i] is the cost of the transition
    leaving state obs[i]; prepend a sentinel 0 so a next-step cost gives delta=1.
    """
    env = safety_gymnasium.make(env_id)
    rng = np.random.default_rng(seed)
    try:
        env.reset(seed=seed)
    except TypeError:
        env.reset()

    all_obs, all_lbl = [], []
    for _ in range(episodes):
        out = env.reset()
        obs = out[0] if isinstance(out, tuple) else out
        ep_obs = [np.asarray(obs, dtype=np.float32)]
        ep_costs: list[float] = []
        done = False
        while not done:
            a = env.action_space.sample()
            step = env.step(a)
            # safety_gymnasium returns (obs, reward, cost, term, trunc, info)
            nobs, _reward, cost, term, trunc, _info = step
            ep_costs.append(float(cost))
            done = bool(term) or bool(trunc)
            if not done:
                ep_obs.append(np.asarray(nobs, dtype=np.float32))
        # Label this episode.
        costs = np.asarray(ep_costs, dtype=np.float64)
        costs_prepended = np.concatenate([[0.0], costs])
        labels_full = label_trajectory(costs_prepended, safety_horizon, bin_size)
        labels = labels_full[: len(costs)]
        n = min(len(ep_obs), len(labels))
        for i in range(n):
            all_obs.append(ep_obs[i])
            all_lbl.append(int(labels[i]))
    env.close()
    return np.asarray(all_obs, dtype=np.float32), np.asarray(all_lbl, dtype=np.int64)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train + save a source S2C (Fig 6).")
    ap.add_argument("--env-id", default="SafetyPointButton1-v0")
    ap.add_argument("--episodes", type=int, default=400,
                    help="random-policy episodes to collect for labeling")
    ap.add_argument("--epochs", type=int, default=30,
                    help="passes over the collected dataset")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--safety-horizon", type=int, default=DEFAULT_SAFETY_HORIZON)
    ap.add_argument("--bin-size", type=int, default=DEFAULT_BIN_SIZE)
    ap.add_argument("--hidden", type=int, nargs="+", default=[64, 64],
                    help="S2C hidden sizes. MUST match the transfer target's S2C "
                         "arch: PPO-Lag (on-policy) uses (64,64), so default is "
                         "(64,64) to make the frozen-load shape-compatible.")
    ap.add_argument("--activation", default="tanh", choices=["relu", "tanh"],
                    help="Match the on-policy S2C activation (tanh) for transfer.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="experiments/transfer/s2c_button1.pt")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"[source-s2c] collecting {args.episodes} episodes on {args.env_id} ...")
    obs, lbl = collect_labeled_data(
        args.env_id, args.episodes, args.safety_horizon, args.bin_size, args.seed
    )
    raw_dim = obs.shape[1]
    n_bins = int(np.ceil(args.safety_horizon / args.bin_size))
    print(f"[source-s2c] collected {len(obs)} labeled states | raw_dim={raw_dim} "
          f"| label range [{lbl.min()}, {lbl.max()}] over {n_bins} bins")
    # Sanity: report the label histogram so we can see cost events were captured.
    hist = np.bincount(lbl, minlength=n_bins)
    print(f"[source-s2c] label histogram (bin: count): "
          f"{ {int(i): int(h) for i, h in enumerate(hist)} }")

    device = torch.device(args.device)
    act = {"relu": torch.nn.ReLU, "tanh": torch.nn.Tanh}[args.activation]
    s2c = S2CModel(
        obs_dim=raw_dim,
        safety_horizon=args.safety_horizon,
        bin_size=args.bin_size,
        hidden_sizes=tuple(args.hidden),
        activation=act,
    ).to(device)
    opt = torch.optim.Adam(s2c.parameters(), lr=args.lr)

    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
    lbl_t = torch.as_tensor(lbl, dtype=torch.long, device=device)
    n = len(obs_t)
    print(f"[source-s2c] training S2C: {args.epochs} epochs, batch {args.batch_size}")
    for ep in range(args.epochs):
        perm = torch.randperm(n, device=device)
        total = 0.0
        nb = 0
        for start in range(0, n, args.batch_size):
            idx = perm[start:start + args.batch_size]
            s2c.train()
            opt.zero_grad()
            loss = s2c.nll_loss(obs_t[idx], lbl_t[idx])
            loss.backward()
            opt.step()
            total += float(loss.item())
            nb += 1
        if ep == 0 or (ep + 1) % 5 == 0 or ep == args.epochs - 1:
            print(f"  epoch {ep + 1:3d}/{args.epochs}  mean NLL {total / max(nb,1):.4f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({
        "s2c_state_dict": s2c.state_dict(),
        "raw_dim": raw_dim,
        "safety_horizon": args.safety_horizon,
        "bin_size": args.bin_size,
        "hidden": list(args.hidden),
        "activation": args.activation,
        "source_env": args.env_id,
    }, args.out)
    print(f"[source-s2c] saved S2C (input dim {raw_dim}) -> {args.out}")
    print(f"[source-s2c] use it for transfer with: "
          f"--load-s2c {args.out} --s2c-input-dim {raw_dim}")


if __name__ == "__main__":
    main()
