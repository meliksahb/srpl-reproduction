"""Training entry point for SRPL reproduction (Stage 2 spike + full runs).

Launches an OmniSafe safe-RL algorithm (PPOLag / TD3Lag / SACLag) on an SRPL
Safety-Gymnasium environment. Importing ``srpl.envs.safety_gym_srpl`` registers
the SRPL env ids with OmniSafe, so we pass e.g. ``SafetyPointGoal1SRPL-v0``
(SR on) or ``SafetyPointGoal1Base-v0`` (baseline).

S2C hyperparameters are chosen by algorithm FAMILY and delivered to the env
wrapper via a module-level config hook (``set_s2c_config``). We use the hook
rather than OmniSafe's ``env_cfgs`` because OmniSafe only forwards ``env_cfgs``
for on-policy algorithms (the off-policy configs reject the key). Settings
follow Appendix A.3.2:

    on-policy  (PPOLag):            S2C lr 1e-5, batch 5000, update_freq 100,
                                    net (64,64) + tanh   (matches PPO-Lag policy)
    off-policy (TD3Lag / SACLag):   S2C lr 1e-3, batch 512, update_freq 20000,
                                    net (256,256) + relu (matches off-policy policy)

The "match the policy network" choice follows the paper's statement that the S2C
"has the same network architecture as the policy".

Examples
--------
Full runs (defaults: PPO-Lag 2M, off-policy 1M):
    python scripts/train.py --algo PPOLag --task PointGoal1 --sr --seed 0
    python scripts/train.py --algo TD3Lag --task PointGoal1 --sr --seed 0

Fast smoke that SHOWS learning in TD3 (lower the warmup so updates start early):
    python scripts/train.py --algo TD3Lag --task PointGoal1 --sr --seed 0 \
        --steps 30000 --start-learning-steps 2000 --logdir ./experiments/smoke
"""

from __future__ import annotations

import argparse
import os

# IMPORTANT: importing this module registers the SRPL env ids with OmniSafe.
import srpl.envs.safety_gym_srpl  # noqa: F401  (registration side effect)
from srpl.envs.safety_gym_srpl import set_s2c_config


# task -> (SR id, baseline id)
_TASK_IDS = {
    "PointGoal1": ("SafetyPointGoal1SRPL-v0", "SafetyPointGoal1Base-v0"),
    "PointButton1": ("SafetyPointButton1SRPL-v0", "SafetyPointButton1Base-v0"),
}

_ON_POLICY = {"PPOLag"}
_OFF_POLICY = {"TD3Lag", "SACLag"}

# S2C hyperparameters by algorithm family (paper Appendix A.3.2 + "match policy").
_S2C_ON_POLICY = dict(
    s2c_lr=1e-5, s2c_batch_size=5000, s2c_update_freq=100,
    s2c_hidden_sizes=(64, 64), s2c_activation="tanh",
)
_S2C_OFF_POLICY = dict(
    s2c_lr=1e-3, s2c_batch_size=512, s2c_update_freq=20000,
    s2c_hidden_sizes=(256, 256), s2c_activation="relu",
)

# Per-algo default training horizon (overridable with --steps for smoke tests).
_DEFAULT_STEPS = {"PPOLag": 2_000_000, "TD3Lag": 1_000_000, "SACLag": 1_000_000}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SRPL reproduction trainer")
    p.add_argument("--algo", default="PPOLag",
                   choices=["PPOLag", "TD3Lag", "SACLag"])
    p.add_argument("--task", default="PointGoal1", choices=list(_TASK_IDS))
    sr = p.add_mutually_exclusive_group()
    sr.add_argument("--sr", dest="sr", action="store_true",
                    help="Use SRPL augmentation (default).")
    sr.add_argument("--no-sr", dest="sr", action="store_false",
                    help="Baseline: no safety augmentation.")
    p.set_defaults(sr=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=None,
                   help="Total env steps. Default: per-algo (PPO 2M, off-policy 1M).")
    p.add_argument("--steps-per-epoch", type=int, default=None,
                   help="Override OmniSafe steps_per_epoch. Default: use algo default "
                        "(on-policy 20000, off-policy 2000). Leave unset normally.")
    p.add_argument("--start-learning-steps", type=int, default=None,
                   help="Off-policy only: random-action warmup before updates. "
                        "Lower it (e.g. 2000) for a fast smoke that shows learning. "
                        "Default: config (TD3 25000, SAC 10000).")
    p.add_argument("--update-cycle", type=int, default=None,
                   help="Off-policy only: env steps collected between update "
                        "bursts. OmniSafe default 1 (update every step) is slow "
                        "on CPU; e.g. 50 batches the work for a large speedup.")
    p.add_argument("--update-iters", type=int, default=None,
                   help="Off-policy only: gradient steps per update burst. Pair "
                        "with --update-cycle to keep the gradient:env-step ratio "
                        "(e.g. --update-cycle 50 --update-iters 50 ~= 1:1).")
    p.add_argument("--cost-limit", type=float, default=10.0,
                   help="Safety-Gym cost threshold beta (paper: 10).")
    # ---- Transfer (Figure 6) ----
    p.add_argument("--frozen-s2c", action="store_true",
                   help="Do not train the S2C (apply it frozen). Used for the "
                        "transfer experiment and its random-init control.")
    p.add_argument("--load-s2c", default=None,
                   help="Path to a source S2C checkpoint to load (implies "
                        "--frozen-s2c). Omit for a frozen RANDOM-init control.")
    p.add_argument("--s2c-input-dim", type=int, default=0,
                   help="Build the S2C for this input dim and pad/truncate the "
                        "raw obs to it (e.g. 76 to apply a PointButton1 S2C to "
                        "PointGoal1, whose 60-dim obs is zero-padded to 76).")
    p.add_argument("--device", default="cpu")
    p.add_argument("--logdir", default="./experiments/spike")
    p.add_argument("--torch-threads", type=int, default=0,
                   help="Cap torch CPU threads (0 = use OmniSafe default 16). "
                        "The parallel launcher sets this to ~2 per run.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    is_off_policy = args.algo in _OFF_POLICY

    if args.torch_threads and args.torch_threads > 0:
        os.environ.setdefault("OMP_NUM_THREADS", str(args.torch_threads))
        import torch
        torch.set_num_threads(args.torch_threads)

    import omnisafe

    sr_id, base_id = _TASK_IDS[args.task]
    env_id = sr_id if args.sr else base_id
    tag = "SR" if args.sr else "BASE"
    total_steps = args.steps if args.steps is not None else _DEFAULT_STEPS[args.algo]

    # S2C config by family. Delivered via the module-level hook (works for both
    # on- and off-policy; OmniSafe only forwards env_cfgs for on-policy).
    s2c_cfg = dict(_S2C_OFF_POLICY if is_off_policy else _S2C_ON_POLICY)
    # Transfer overrides (Figure 6): frozen / loaded / padded S2C.
    if args.frozen_s2c or args.load_s2c is not None:
        s2c_cfg["frozen_s2c"] = True
    if args.load_s2c is not None:
        s2c_cfg["load_s2c_path"] = args.load_s2c
    if args.s2c_input_dim and args.s2c_input_dim > 0:
        s2c_cfg["s2c_input_dim"] = args.s2c_input_dim
    set_s2c_config(**s2c_cfg)

    algo_cfgs: dict = {}
    if args.steps_per_epoch is not None:
        algo_cfgs["steps_per_epoch"] = args.steps_per_epoch
    if is_off_policy and args.start_learning_steps is not None:
        algo_cfgs["start_learning_steps"] = args.start_learning_steps
    if is_off_policy and args.update_cycle is not None:
        algo_cfgs["update_cycle"] = args.update_cycle
    if is_off_policy and args.update_iters is not None:
        algo_cfgs["update_iters"] = args.update_iters

    custom_cfgs = {
        "seed": args.seed,
        "train_cfgs": {
            "total_steps": total_steps,
            "vector_env_nums": 1,     # SRPL env supports num_envs=1
            "parallel": 1,
            "device": args.device,
        },
        "lagrange_cfgs": {
            "cost_limit": args.cost_limit,
        },
        "logger_cfgs": {
            "use_tensorboard": True,
            "use_wandb": False,
            "log_dir": args.logdir,
        },
    }
    if args.torch_threads and args.torch_threads > 0:
        custom_cfgs["train_cfgs"]["torch_threads"] = args.torch_threads
    if algo_cfgs:
        custom_cfgs["algo_cfgs"] = algo_cfgs

    print(f"[train] algo={args.algo} env={env_id} ({tag}) seed={args.seed} "
          f"steps={total_steps} device={args.device} "
          f"family={'off-policy' if is_off_policy else 'on-policy'}")

    agent = omnisafe.Agent(args.algo, env_id, custom_cfgs=custom_cfgs)
    agent.learn()
    print(f"[train] DONE: {args.algo} {env_id} seed={args.seed}")


if __name__ == "__main__":
    main()
