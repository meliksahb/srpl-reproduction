"""Figure 1 reproduction: DQN vs DQN + ground-truth safety on Island Navigation.

Trains two agents on the Island Navigation gridworld and compares them:
    * BASE: vanilla DQN on the one-hot position (safety slot zero-filled).
    * GT  : DQN whose observation is augmented with the GROUND-TRUTH distance-to
            -water one-hot (the oracle analogue of the learned S2C output).

Both agents are identical except for whether the safety slot carries real
information. The SRPL paper's Figure 1 argues that ground-truth safety
information improves value estimation and reduces conservativeness; the concrete,
measurable manifestation here is that the GT agent learns a safe goal-reaching
policy faster and accumulates fewer unsafe (water) transitions during training.

This is a standalone toy: pure PyTorch + numpy, no OmniSafe/Safety-Gymnasium.
It runs in well under a minute for the default settings.

Run:
    python island_navigation/dqn.py --layout island --episodes 400 --seeds 5
Outputs fig1_island.png (+ a short printed summary) into --out (default: cwd).
"""

from __future__ import annotations

import argparse
import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn

from island_navigation.env import IslandNavigation


class QNet(nn.Module):
    """Small MLP Q-network: obs -> Q-value per action."""

    def __init__(self, obs_dim: int, n_actions: int, hidden=(64, 64)):
        super().__init__()
        layers, last = [], obs_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers.append(nn.Linear(last, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Replay:
    """Minimal uniform replay buffer."""

    def __init__(self, cap: int):
        self.buf: deque = deque(maxlen=cap)

    def push(self, *t):
        self.buf.append(t)

    def sample(self, n: int):
        batch = random.sample(self.buf, n)
        s, a, r, s2, d = zip(*batch)
        return (torch.as_tensor(np.array(s), dtype=torch.float32),
                torch.as_tensor(a, dtype=torch.long),
                torch.as_tensor(r, dtype=torch.float32),
                torch.as_tensor(np.array(s2), dtype=torch.float32),
                torch.as_tensor(d, dtype=torch.float32))

    def __len__(self):
        return len(self.buf)


def train_one(
    use_safety: bool,
    layout: str = "island",
    episodes: int = 400,
    seed: int = 0,
    gamma: float = 0.95,
    lr: float = 1e-3,
    buffer_cap: int = 10_000,
    batch_size: int = 64,
    target_update: int = 200,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    eps_decay_frac: float = 0.5,
    shaping_coef: float = 1.0,
):
    """Train one DQN variant; return per-episode (return, cost, success) arrays.

    Potential-based reward shaping (Ng et al., 1999) with potential
    Phi(s) = -manhattan_distance_to_goal guides exploration to the goal so DQN
    learns reliably on this sparse-reward toy. Shaping is policy-invariant (it
    cannot change the optimal policy) and is applied ONLY to the reward stored
    in the replay buffer; the reported episodic return uses the TRUE unshaped
    reward, and cost is never shaped. With reliable goal-reaching, the remaining
    difference between arms is whether the agent knows where water is -- which is
    precisely the safety signal under test.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = IslandNavigation(layout=layout, use_safety=use_safety)
    gr, gc = env._goal

    def potential(cell) -> float:
        return -(abs(cell[0] - gr) + abs(cell[1] - gc))

    q = QNet(env.obs_dim, env.num_actions)
    qt = QNet(env.obs_dim, env.num_actions)
    qt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=lr)
    buf = Replay(buffer_cap)

    eps_decay_episodes = max(1, int(eps_decay_frac * episodes))
    global_step = 0
    rets, costs, succ = [], [], []

    for ep in range(episodes):
        frac = min(1.0, ep / eps_decay_episodes)
        eps = eps_start + frac * (eps_end - eps_start)
        s = env.reset()
        ep_ret = ep_cost = 0.0
        done = False
        reached = False
        while not done:
            cell_before = env._agent
            if random.random() < eps:
                a = random.randrange(env.num_actions)
            else:
                with torch.no_grad():
                    a = int(q(torch.as_tensor(s, dtype=torch.float32)).argmax())
            s2, r, c, term, trunc, info = env.step(a)
            done = term or trunc
            # Potential-based shaping: F = gamma*Phi(s') - Phi(s); Phi=0 at terminal.
            phi_s = potential(cell_before)
            phi_s2 = 0.0 if term else potential(env._agent)
            shaped_r = r + shaping_coef * (gamma * phi_s2 - phi_s)
            buf.push(s, a, shaped_r, s2, float(term))
            s = s2
            ep_ret += r           # TRUE (unshaped) return for honest reporting
            ep_cost += c
            reached = reached or info["success"]
            global_step += 1

            if len(buf) >= batch_size:
                bs, ba, br, bs2, bd = buf.sample(batch_size)
                qvals = q(bs).gather(1, ba.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    target = br + gamma * (1 - bd) * qt(bs2).max(dim=1).values
                loss = nn.functional.smooth_l1_loss(qvals, target)
                opt.zero_grad()
                loss.backward()
                opt.step()
                if global_step % target_update == 0:
                    qt.load_state_dict(q.state_dict())

        rets.append(ep_ret)
        costs.append(ep_cost)
        succ.append(1.0 if reached else 0.0)

    return np.array(rets), np.array(costs), np.array(succ)


def _smooth(x: np.ndarray, k: int = 15) -> np.ndarray:
    """Centered moving average for readable curves."""
    if k <= 1 or len(x) < k:
        return x
    ker = np.ones(k) / k
    return np.convolve(x, ker, mode="same")


def run_comparison(layout: str, episodes: int, seeds: int, out: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = {}  # arm -> (rets, costs, succ) stacked over seeds
    for arm, use_safety in [("DQN (base)", False), ("DQN + GT-safety", True)]:
        R, C, S = [], [], []
        for sd in range(seeds):
            r, c, s = train_one(use_safety=use_safety, layout=layout,
                                 episodes=episodes, seed=sd)
            R.append(r); C.append(c); S.append(s)
            print(f"  {arm:18s} seed {sd}: "
                  f"final return {r[-20:].mean():7.2f} | "
                  f"final cost {c[-20:].mean():.3f} | "
                  f"final success {s[-20:].mean():.2f} | "
                  f"cumulative water hits {int(c.sum())}")
        results[arm] = (np.vstack(R), np.vstack(C), np.vstack(S))

    colors = {"DQN (base)": "#888888", "DQN + GT-safety": "#1f77b4"}
    fig, (ax_r, ax_c) = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(episodes)
    for arm, (R, C, _S) in results.items():
        rm, rs = R.mean(0), R.std(0)
        cm, cs = C.mean(0), C.std(0)
        ax_r.plot(x, _smooth(rm), color=colors[arm], label=arm)
        ax_r.fill_between(x, _smooth(rm - rs), _smooth(rm + rs),
                          color=colors[arm], alpha=0.15)
        ax_c.plot(x, _smooth(cm), color=colors[arm], label=arm)
        ax_c.fill_between(x, _smooth(cm - cs), _smooth(cm + cs),
                          color=colors[arm], alpha=0.15)
    ax_r.set_title(f"Island Navigation ({layout}): Episodic Return")
    ax_r.set_xlabel("episode"); ax_r.set_ylabel("return")
    ax_r.legend(); ax_r.grid(alpha=0.3)
    ax_c.set_title(f"Island Navigation ({layout}): Episodic Cost (water hits)")
    ax_c.set_xlabel("episode"); ax_c.set_ylabel("cost per episode")
    ax_c.legend(); ax_c.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"fig1_{layout}.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)

    # Printed summary (the headline numbers for the README).
    print("\n=== Figure 1 summary (mean over seeds) ===")
    for arm, (R, C, S) in results.items():
        print(f"{arm:18s}: final return {R[:, -20:].mean():7.2f} | "
              f"final cost {C[:, -20:].mean():.3f} | "
              f"final success {S[:, -20:].mean():.2f} | "
              f"cumulative training water hits {C.sum(1).mean():.0f}")
    print(f"\nwrote {path}")
    return path


def main():
    ap = argparse.ArgumentParser(description="Figure 1: DQN vs DQN+GT-safety")
    ap.add_argument("--layout", default="pond", choices=["island", "pond"],
                    help="pond is the Figure-1 layout (internal water forces "
                         "near-water routing, where GT safety clearly helps).")
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", default=".")
    args = ap.parse_args()
    run_comparison(args.layout, args.episodes, args.seeds, args.out)


if __name__ == "__main__":
    main()
