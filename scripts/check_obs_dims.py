"""Check observation dimensionality for the transfer experiment (Figure 6).

The Figure 6 transfer trains an S2C model on PointButton1, freezes it, and
applies it to PointGoal1. For a frozen S2C to accept PointGoal1 observations,
the *raw* observation dimension (the S2C's input) must be identical across the
two tasks. The paper had to build a unified-LiDAR wrapper (Appendix A.4) because
in raw Safety-Gym the two tasks differ (PointButton1 has extra object types ->
extra LiDAR groups). This script checks whether that mismatch still exists in
your installed safety-gymnasium version.

It prints, for each task:
  * raw Safety-Gymnasium observation dim  (this is what the S2C sees)
  * our SRPL-wrapped observation dim       (raw + safety_horizon augmentation)

Interpretation:
  * If the two RAW dims are EQUAL  -> transfer is simple: the frozen S2C trained
        on PointButton1 ingests PointGoal1 obs directly. No wrapper needed.
  * If the two RAW dims DIFFER     -> we must replicate a unified-obs wrapper
        (pad/truncate/aggregate to a common dim) before training/applying S2C.

Run:
    conda activate srpl && unset PYTHONPATH && cd ~/srpl-reproduction
    PYTHONPATH=. python scripts/check_obs_dims.py
"""

from __future__ import annotations

import safety_gymnasium


RAW_TASKS = ["SafetyPointGoal1-v0", "SafetyPointButton1-v0"]


def raw_obs_dim(env_id: str) -> int:
    env = safety_gymnasium.make(env_id)
    try:
        space = env.observation_space
        dim = int(space.shape[0])
    finally:
        env.close()
    return dim


def main() -> None:
    print("=" * 64)
    print(" Observation-dimension check for the Fig 6 transfer experiment")
    print("=" * 64)

    raw = {}
    for tid in RAW_TASKS:
        try:
            raw[tid] = raw_obs_dim(tid)
            print(f"  raw  {tid:28s} obs_dim = {raw[tid]}")
        except Exception as e:  # noqa: BLE001
            print(f"  raw  {tid:28s} ERROR: {e}")

    # Our SRPL-wrapped envs advertise raw + safety_horizon (default Hs/bin -> 20).
    # Report the augmented dims too, for completeness.
    try:
        from srpl.envs.safety_gym_srpl import SafetyGymSRPLEnv  # noqa: F401
        print("\n  (SRPL-wrapped envs add the safety representation on top of raw:")
        print("   augmented_dim = raw_dim + num_bins, with default num_bins = 20)")
    except Exception as e:  # noqa: BLE001
        print(f"\n  (could not import SRPL wrapper for augmented dims: {e})")

    print("\n" + "-" * 64)
    if len(raw) == 2:
        a, b = raw[RAW_TASKS[0]], raw[RAW_TASKS[1]]
        if a == b:
            print(f"  RESULT: raw dims MATCH ({a} == {b}).")
            print("  -> Transfer is simple: a frozen S2C trained on PointButton1")
            print("     ingests PointGoal1 observations directly. No unified-obs")
            print("     wrapper needed.")
        else:
            print(f"  RESULT: raw dims DIFFER (PointGoal1={a}, PointButton1={b}).")
            print("  -> We must replicate a unified-observation scheme (pad to a")
            print("     common dim, or aggregate LiDAR groups) so the frozen S2C")
            print("     input matches across tasks. This mirrors the paper's")
            print("     Appendix A.4 unified-LiDAR wrapper.")
    else:
        print("  RESULT: could not read both tasks; see errors above.")
    print("-" * 64)


if __name__ == "__main__":
    main()
