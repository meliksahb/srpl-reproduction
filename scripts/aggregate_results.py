"""Aggregate SRPL training runs into Figure 4 (curves) and Table 1 (markdown).

Walks an experiments tree of OmniSafe runs, groups them by (task, algorithm,
arm), averages over seeds, and produces:

  * Figure 4 (partial): for each (task, algorithm), a two-panel plot of episodic
    RETURN and episodic COST versus environment steps, comparing the baseline
    against the SRPL-augmented agent (mean +/- std across seeds).
  * Table 1 (corresponding rows): end-of-training average return and the
    training cost-rate (x1e2), per condition.

Run, after the full batch finishes:
    python scripts/aggregate_results.py --base ./experiments/full

Outputs land in <base>/_analysis/ : one PNG per (task, algo) plus table1.md and
a results.csv with the raw aggregated numbers.

Definitions
-----------
* Episodic return / cost: the per-episode Metrics/EpRet and Metrics/EpCost that
  OmniSafe logs every epoch (present for both on- and off-policy algorithms).
* Cost-rate (x1e2): the paper reports cumulative cost over training divided by
  total steps. With uniform episode length L this equals
  mean_over_training(EpCost) / L * 100, i.e. average cost incurred per step
  during learning, scaled by 100. We compute it over ALL logged epochs (it is a
  "how unsafe was the agent while learning" metric, not an end-of-training one).
* End-of-training return/cost: mean over the final 10% of epochs (>=1).
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Run dir parent looks like:  PPOLag-{SafetyPointGoal1SRPL-v0}
_EXP_RE = re.compile(r"^(?P<algo>[A-Za-z0-9]+)-\{(?P<envid>[^}]+)\}$")
# env id looks like:  SafetyPointGoal1SRPL-v0  /  SafetyPointButton1Base-v0
_ENV_RE = re.compile(r"^Safety(?P<task>.+?)(?P<arm>SRPL|Base)-v\d+$")
_SEED_RE = re.compile(r"seed-(?P<seed>\d+)")

RET_COL = "Metrics/EpRet"
COST_COL = "Metrics/EpCost"
LEN_COL = "Metrics/EpLen"
STEP_COL = "TotalEnvSteps"

ALGO_ORDER = ["PPOLag", "TD3Lag", "SACLag"]
TASK_ORDER = ["PointGoal1", "PointButton1"]


def find_runs(base: str) -> list[dict]:
    """Locate every progress.csv and parse its (task, algo, arm, seed)."""
    runs = []
    for csv_path in glob.glob(os.path.join(base, "*", "seed-*", "progress.csv")):
        seed_dir = os.path.basename(os.path.dirname(csv_path))
        exp_dir = os.path.basename(os.path.dirname(os.path.dirname(csv_path)))
        m_exp = _EXP_RE.match(exp_dir)
        m_seed = _SEED_RE.search(seed_dir)
        if not m_exp or not m_seed:
            continue
        algo = m_exp.group("algo")
        envid = m_exp.group("envid")
        m_env = _ENV_RE.match(envid)
        if not m_env:
            continue
        runs.append({
            "path": csv_path,
            "algo": algo,
            "task": m_env.group("task"),
            "arm": "SR" if m_env.group("arm") == "SRPL" else "Base",
            "seed": int(m_seed.group("seed")),
        })
    return runs


def load_curve(path: str) -> pd.DataFrame | None:
    """Load a run's progress.csv with the columns we need."""
    try:
        df = pd.read_csv(path)
    except Exception as e:  # noqa: BLE001
        print(f"  ! could not read {path}: {e}")
        return None
    needed = {RET_COL, COST_COL, STEP_COL}
    if not needed.issubset(df.columns):
        print(f"  ! {path} missing columns {needed - set(df.columns)}")
        return None
    return df


def _stack_over_seeds(curves: list[pd.DataFrame], col: str):
    """Align seed curves by epoch index (truncate to shortest) and stack.

    Runs of the same algorithm share steps_per_epoch/total_steps, so the
    TotalEnvSteps grid matches row-for-row; truncating to the shortest length
    is robust to a crashed/short seed.
    Returns (steps, mean, std) or None.
    """
    curves = [c for c in curves if c is not None and len(c) > 0]
    if not curves:
        return None
    n = min(len(c) for c in curves)
    if n == 0:
        return None
    steps = curves[0][STEP_COL].to_numpy()[:n]
    mat = np.vstack([c[col].to_numpy()[:n] for c in curves])  # (seeds, n)
    return steps, mat.mean(axis=0), mat.std(axis=0)


def _final_mean(curve: pd.DataFrame, col: str, frac: float = 0.1) -> float:
    """Mean of `col` over the final `frac` of epochs (at least 1)."""
    k = max(1, int(round(frac * len(curve))))
    return float(curve[col].to_numpy()[-k:].mean())


def _cost_rate(curve: pd.DataFrame) -> float:
    """Training cost-rate (x1e2) = mean_over_training(EpCost) / EpLen * 100."""
    ep_len = 1000.0
    if LEN_COL in curve.columns and curve[LEN_COL].notna().any():
        v = float(curve[LEN_COL].to_numpy()[-1])
        if v > 0:
            ep_len = v
    return float(curve[COST_COL].to_numpy().mean() / ep_len * 100.0)


def make_figure(task: str, algo: str, arms: dict, out_dir: str) -> str | None:
    """Two-panel (return, cost) figure for one (task, algo): SR vs Base."""
    colors = {"Base": "#888888", "SR": "#1f77b4"}
    fig, (ax_r, ax_c) = plt.subplots(1, 2, figsize=(11, 4.2))
    plotted = False
    for arm in ("Base", "SR"):
        if arm not in arms:
            continue
        ret = _stack_over_seeds(arms[arm], RET_COL)
        cost = _stack_over_seeds(arms[arm], COST_COL)
        if ret is None or cost is None:
            continue
        plotted = True
        n_seeds = len(arms[arm])
        s, m, sd = ret
        ax_r.plot(s, m, color=colors[arm], label=f"{arm} (n={n_seeds})")
        ax_r.fill_between(s, m - sd, m + sd, color=colors[arm], alpha=0.2)
        s, m, sd = cost
        ax_c.plot(s, m, color=colors[arm], label=f"{arm} (n={n_seeds})")
        ax_c.fill_between(s, m - sd, m + sd, color=colors[arm], alpha=0.2)
    if not plotted:
        plt.close(fig)
        return None
    ax_r.set_title(f"{algo} on Safety{task}: Episodic Return")
    ax_r.set_xlabel("environment steps"); ax_r.set_ylabel("episodic return")
    ax_r.legend(); ax_r.grid(alpha=0.3)
    ax_c.set_title(f"{algo} on Safety{task}: Episodic Cost")
    ax_c.set_xlabel("environment steps"); ax_c.set_ylabel("episodic cost")
    ax_c.legend(); ax_c.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(out_dir, f"fig4_{task}_{algo}.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate SRPL runs -> Fig4 + Table1")
    ap.add_argument("--base", default="./experiments/full",
                    help="experiments dir containing the run folders")
    ap.add_argument("--final-frac", type=float, default=0.1,
                    help="fraction of final epochs for end-of-training averages")
    args = ap.parse_args()

    out_dir = os.path.join(args.base, "_analysis")
    os.makedirs(out_dir, exist_ok=True)

    runs = find_runs(args.base)
    if not runs:
        print(f"No runs found under {args.base} "
              f"(expected <base>/ALGO-{{ENVID}}/seed-NNN-*/progress.csv).")
        return
    print(f"Found {len(runs)} runs.")

    # group[(task, algo)][arm] -> list of curves
    group: dict = defaultdict(lambda: defaultdict(list))
    # rows for Table 1: keyed (task, algo, arm) -> list of per-seed dicts
    table_rows: dict = defaultdict(list)

    for r in runs:
        df = load_curve(r["path"])
        if df is None:
            continue
        group[(r["task"], r["algo"])][r["arm"]].append(df)
        table_rows[(r["task"], r["algo"], r["arm"])].append({
            "seed": r["seed"],
            "final_return": _final_mean(df, RET_COL, args.final_frac),
            "final_cost": _final_mean(df, COST_COL, args.final_frac),
            "cost_rate_x1e2": _cost_rate(df),
        })

    # ---- Figure 4 ----
    figs = []
    def _ord(key, order):
        return order.index(key) if key in order else len(order)
    for (task, algo) in sorted(group, key=lambda k: (_ord(k[0], TASK_ORDER),
                                                      _ord(k[1], ALGO_ORDER))):
        path = make_figure(task, algo, group[(task, algo)], out_dir)
        if path:
            figs.append(path)
            print(f"  wrote {os.path.relpath(path)}")

    # ---- Table 1 + results.csv ----
    rows = []
    for (task, algo, arm) in sorted(table_rows,
                                    key=lambda k: (_ord(k[0], TASK_ORDER),
                                                   _ord(k[1], ALGO_ORDER),
                                                   k[2])):
        seeds = table_rows[(task, algo, arm)]
        ret = np.array([s["final_return"] for s in seeds])
        cost = np.array([s["final_cost"] for s in seeds])
        crate = np.array([s["cost_rate_x1e2"] for s in seeds])
        rows.append({
            "task": task, "algo": algo, "arm": arm, "n_seeds": len(seeds),
            "return_mean": ret.mean(), "return_std": ret.std(),
            "cost_mean": cost.mean(), "cost_std": cost.std(),
            "cost_rate_mean": crate.mean(), "cost_rate_std": crate.std(),
        })
    res = pd.DataFrame(rows)
    res_csv = os.path.join(out_dir, "results.csv")
    res.to_csv(res_csv, index=False)
    print(f"  wrote {os.path.relpath(res_csv)}")

    # Markdown Table 1
    md = ["# Table 1 (reproduction): end-of-training return and training cost-rate",
          "",
          f"Final-{int(args.final_frac*100)}%-of-training averages; mean +/- std over seeds. "
          "Cost-rate is x1e2 (cumulative-cost / steps). Lower cost is better.",
          "",
          "| Task | Algorithm | Variant | Seeds | Return (mean +/- std) | "
          "Cost (mean +/- std) | Cost-rate x1e2 |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(
            f"| {r['task']} | {r['algo']} | {r['arm']} | {r['n_seeds']} | "
            f"{r['return_mean']:.2f} +/- {r['return_std']:.2f} | "
            f"{r['cost_mean']:.1f} +/- {r['cost_std']:.1f} | "
            f"{r['cost_rate_mean']:.2f} +/- {r['cost_rate_std']:.2f} |"
        )
    # Paired SR-vs-Base deltas (the SRPL effect), where both arms exist.
    md += ["", "## SRPL effect (SR - Base), per task/algorithm", "",
           "| Task | Algorithm | dReturn | dCost | dCost-rate |",
           "|---|---|---|---|---|"]
    by_cond = {(r["task"], r["algo"], r["arm"]): r for r in rows}
    seen = sorted({(r["task"], r["algo"]) for r in rows},
                  key=lambda k: (_ord(k[0], TASK_ORDER), _ord(k[1], ALGO_ORDER)))
    for (task, algo) in seen:
        b = by_cond.get((task, algo, "Base"))
        s = by_cond.get((task, algo, "SR"))
        if b and s:
            md.append(
                f"| {task} | {algo} | {s['return_mean']-b['return_mean']:+.2f} | "
                f"{s['cost_mean']-b['cost_mean']:+.1f} | "
                f"{s['cost_rate_mean']-b['cost_rate_mean']:+.2f} |"
            )
    table_md = os.path.join(out_dir, "table1.md")
    with open(table_md, "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"  wrote {os.path.relpath(table_md)}")

    print(f"\nDone. {len(figs)} figures + table1.md + results.csv in {out_dir}")


if __name__ == "__main__":
    main()
