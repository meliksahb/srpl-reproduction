"""Plot the Figure 6 transfer experiment: frozen-transfer vs random vs base.

Reads the three target-condition run trees produced by run_transfer.sh:
    <base>/transfer/<...>/seed-*/progress.csv   (frozen PointButton1 S2C)
    <base>/random/<...>/seed-*/progress.csv     (frozen random-init S2C)
    <base>/base/<...>/seed-*/progress.csv        (vanilla PPO-Lag)
and produces a two-panel plot (episodic return + episodic cost vs steps), each
condition averaged over seeds (mean +/- std), mirroring the paper's Figure 6
(left): zero-shot transfer of a frozen safety representation should improve
sample efficiency over the no-information baselines.

Run:
    python scripts/plot_transfer.py --base ./experiments/transfer
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


RET_COL = "Metrics/EpRet"
COST_COL = "Metrics/EpCost"
STEP_COL = "TotalEnvSteps"

CONDITIONS = [
    ("transfer", "SR-PPO-Lag (frozen Button1 S2C)", "#1f77b4"),
    ("random",   "SR-PPO-Lag (frozen random S2C)",  "#ff7f0e"),
    ("base",     "PPO-Lag (no S2C)",                "#888888"),
]


def load_condition(base: str, cond: str):
    """Stack EpRet/EpCost over seeds for one condition; align by shortest."""
    paths = sorted(glob.glob(os.path.join(base, cond, "*", "seed-*", "progress.csv")))
    rets, costs, steps_ref = [], [], None
    for p in paths:
        try:
            df = pd.read_csv(p)
        except Exception:  # noqa: BLE001
            continue
        if not {RET_COL, COST_COL, STEP_COL}.issubset(df.columns):
            continue
        rets.append(df[RET_COL].to_numpy())
        costs.append(df[COST_COL].to_numpy())
        if steps_ref is None or len(df) < len(steps_ref):
            steps_ref = df[STEP_COL].to_numpy()
    if not rets:
        return None
    n = min(min(len(r) for r in rets), len(steps_ref))
    steps = steps_ref[:n]
    R = np.vstack([r[:n] for r in rets])
    C = np.vstack([c[:n] for c in costs])
    return steps, R, C, len(rets)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot Fig 6 transfer results")
    ap.add_argument("--base", default="./experiments/transfer")
    ap.add_argument("--cost-limit", type=float, default=10.0)
    args = ap.parse_args()

    out_dir = os.path.join(args.base, "_analysis")
    os.makedirs(out_dir, exist_ok=True)

    fig, (ax_r, ax_c) = plt.subplots(1, 2, figsize=(11, 4.2))
    found = []
    for cond, label, color in CONDITIONS:
        res = load_condition(args.base, cond)
        if res is None:
            print(f"  (no data for condition '{cond}')")
            continue
        steps, R, C, nseed = res
        found.append((cond, label, R, C, nseed))
        rm, rs = R.mean(0), R.std(0)
        cm, cs = C.mean(0), C.std(0)
        ax_r.plot(steps, rm, color=color, label=f"{label} (n={nseed})")
        ax_r.fill_between(steps, rm - rs, rm + rs, color=color, alpha=0.15)
        ax_c.plot(steps, cm, color=color, label=f"{label} (n={nseed})")
        ax_c.fill_between(steps, cm - cs, cm + cs, color=color, alpha=0.15)

    if not found:
        print(f"No transfer runs found under {args.base}. Run run_transfer.sh first.")
        return

    ax_c.axhline(args.cost_limit, ls="--", color="black", lw=1, label="cost limit")
    ax_r.set_title("Transfer (PointButton1 -> PointGoal1): Episodic Return")
    ax_r.set_xlabel("environment steps"); ax_r.set_ylabel("episodic return")
    ax_r.legend(fontsize=8); ax_r.grid(alpha=0.3)
    ax_c.set_title("Transfer (PointButton1 -> PointGoal1): Episodic Cost")
    ax_c.set_xlabel("environment steps"); ax_c.set_ylabel("episodic cost")
    ax_c.legend(fontsize=8); ax_c.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "fig6_transfer.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)

    # End-of-training summary.
    print("\n=== Figure 6 transfer summary (final 10% of training) ===")
    for cond, label, R, C, nseed in found:
        k = max(1, int(0.1 * R.shape[1]))
        print(f"{label:38s} (n={nseed}): "
              f"return {R[:, -k:].mean():6.2f} +/- {R[:, -k:].mean(1).std():.2f} | "
              f"cost {C[:, -k:].mean():6.2f} +/- {C[:, -k:].mean(1).std():.2f}")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
