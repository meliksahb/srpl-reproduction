"""Env-level smoke test for the SRPL Safety-Gymnasium wrapper.

Runs BEFORE the full training spike to isolate environment-wrapper issues from
OmniSafe training-config issues. It instantiates the wrapper directly (bypassing
OmniSafe's Agent/adapter), checks the augmented observation space, steps through
a couple of episodes, and confirms the S2C model actually trains.

Run:
    python scripts/env_check.py
Expected: augmented obs dim 80 (= 60 + 20), episodes flush, S2C loss decreases,
and the baseline env's safety slot is constant while the SR env's varies.
"""

from __future__ import annotations

import numpy as np
import torch

from srpl.envs.safety_gym_srpl import SafetyGymSRPLEnv


def _act(env) -> torch.Tensor:
    return torch.as_tensor(env.action_space.sample(), dtype=torch.float32)


def check_sr() -> None:
    print("=" * 64)
    print("SR env: SafetyPointGoal1SRPL-v0")
    print("=" * 64)
    # Small warmup/batch so the S2C trains within a short check.
    env = SafetyGymSRPLEnv(
        "SafetyPointGoal1SRPL-v0",
        num_envs=1,
        device="cpu",
        s2c_warmup=400,
        s2c_batch_size=256,
        s2c_update_freq=50,
    )
    print("observation_space:", env.observation_space)   # expect Box(80,)
    print("action_space:", env.action_space)             # expect Box(2,)
    assert env.observation_space.shape == (80,), "augmented obs should be 80-dim"

    obs, _ = env.reset(seed=0)
    assert obs.shape == (80,), f"reset obs should be (80,), got {tuple(obs.shape)}"

    n_steps = 2300        # > 2 episodes (PointGoal1 truncates at 1000) + warmup
    episodes = 0
    safety_slots = []
    for t in range(n_steps):
        obs, reward, cost, terminated, truncated, info = env.step(_act(env))
        assert obs.shape == (80,)
        safety_slots.append(obs[60:].detach().numpy().copy())
        if bool(terminated.item()) or bool(truncated.item()):
            episodes += 1

    slots = np.stack(safety_slots)
    print(f"stepped {n_steps} steps | episodes finished: {episodes}")
    print(f"S2C updates performed : {env._s2c_updates}")
    print(f"S2C last loss         : {env._s2c_loss_last:.4f}")
    print(f"S2C buffer size       : {len(env._buf_obs)}")
    print(f"safety slot row sum    (should be ~1.0): {slots[-1].sum():.4f}")
    print(f"safety slot variance over time (should be > 0): {slots.var():.6f}")

    assert episodes >= 1, "no episode finished — labeling never triggered"
    assert env._s2c_updates > 0, "S2C never trained"
    assert slots.var() > 0, "safety slot is constant — S2C output not varying"
    # Distribution sanity: each safety slot row should sum to ~1 (softmax).
    assert abs(slots[-1].sum() - 1.0) < 1e-3, "safety slot is not a distribution"
    print("SR env: OK\n")
    env.close()


def check_baseline() -> None:
    print("=" * 64)
    print("Baseline env: SafetyPointGoal1Base-v0")
    print("=" * 64)
    env = SafetyGymSRPLEnv("SafetyPointGoal1Base-v0", num_envs=1, device="cpu")
    print("observation_space:", env.observation_space)
    assert env.observation_space.shape == (80,), "baseline obs should also be 80-dim"

    obs, _ = env.reset(seed=0)
    slots = []
    for _ in range(50):
        obs, *_ = env.step(_act(env))
        slots.append(obs[60:].detach().numpy().copy())
    slots = np.stack(slots)
    print(f"safety slot variance (should be 0): {slots.var():.6f}")
    print(f"S2C updates (should be 0)         : {env._s2c_updates}")
    assert slots.var() == 0.0, "baseline safety slot should be constant"
    assert env._s2c_updates == 0, "baseline must not train the S2C"
    print("Baseline env: OK\n")
    env.close()


if __name__ == "__main__":
    check_sr()
    check_baseline()
    print("ALL ENV CHECKS PASSED")
