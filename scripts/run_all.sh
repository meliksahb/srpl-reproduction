#!/usr/bin/env bash
# Parallel launcher for the full SRPL reproduction grid.
#
# Runs the whole experiment as a concurrency-limited queue. Each run is capped
# to a small number of torch threads (the key finding: small nets train FASTER
# with few threads, and N jobs x 2 threads parallelize cleanly on this box).
#
# Grid (3 seeds each, base + SR):
#   PPO-Lag   : PointGoal1, PointButton1   @ 1,000,000 steps   (headline; runs FIRST)
#   TD3-Lag   : PointGoal1, PointButton1   @   500,000 steps
#   SAC-Lag   : PointGoal1, PointButton1   @   500,000 steps
#   => 12 + 12 + 12 = 36 runs.
#
# Resumable: a per-run marker file is written on success; re-running the script
# skips completed runs. Safe to Ctrl-C and restart (a crashed run just reruns).
#
# Usage:
#   conda activate srpl && unset PYTHONPATH && cd ~/srpl-reproduction
#   bash scripts/run_all.sh                 # full grid, 4 parallel
#   MAX_PAR=6 bash scripts/run_all.sh       # override parallelism
#   SEEDS="0 1" bash scripts/run_all.sh     # fewer seeds
#   OFFPOLICY_TASKS="PointGoal1" bash scripts/run_all.sh   # off-policy: 1 env only (fallback)
#
# Monitor progress from another shell:
#   tail -f experiments/full/_logs/PPOLag_PointGoal1_sr_seed0.log
#   ls experiments/full/_markers/                 # which runs are done

set -uo pipefail

# ----------------------------- configuration ------------------------------ #
MAX_PAR="${MAX_PAR:-4}"                 # concurrent runs (4 = 8 threads = 8 phys cores)
THREADS="${THREADS:-2}"                 # torch threads per run
SEEDS="${SEEDS:-0 1 2}"                 # random seeds
PPO_STEPS="${PPO_STEPS:-1000000}"       # on-policy horizon
OFF_STEPS="${OFF_STEPS:-500000}"        # off-policy horizon
PPO_TASKS="${PPO_TASKS:-PointGoal1 PointButton1}"
OFFPOLICY_TASKS="${OFFPOLICY_TASKS:-PointGoal1 PointButton1}"
OFF_EXTRA="${OFF_EXTRA:---update-cycle 50 --update-iters 50}"  # 1:1 ratio, batched
BASE="${BASE:-./experiments/full}"
PY="${PY:-python}"

MARK="$BASE/_markers"; LOGS="$BASE/_logs"
mkdir -p "$MARK" "$LOGS"

echo "=================================================================="
echo " SRPL full run | MAX_PAR=$MAX_PAR THREADS=$THREADS SEEDS='$SEEDS'"
echo " PPO: $PPO_STEPS steps, tasks: $PPO_TASKS"
echo " OFF: $OFF_STEPS steps, tasks: $OFFPOLICY_TASKS, extra: $OFF_EXTRA"
echo " logs -> $LOGS   markers -> $MARK"
echo "=================================================================="

# ------------------------------ build queue ------------------------------- #
# Each entry: "algo|task|arm|seed|steps|extra"
JOBS=()
# PPO-Lag FIRST (headline, fast — guarantees the must-have completes early).
for task in $PPO_TASKS; do
  for arm in sr no-sr; do
    for s in $SEEDS; do
      JOBS+=("PPOLag|$task|$arm|$s|$PPO_STEPS|")
    done
  done
done
# Off-policy next (slower).
for algo in TD3Lag SACLag; do
  for task in $OFFPOLICY_TASKS; do
    for arm in sr no-sr; do
      for s in $SEEDS; do
        JOBS+=("$algo|$task|$arm|$s|$OFF_STEPS|$OFF_EXTRA")
      done
    done
  done
done

echo "Queued ${#JOBS[@]} runs."

# ------------------------------- run one ---------------------------------- #
run_one() {
  local spec="$1"
  local algo task arm seed steps extra
  IFS='|' read -r algo task arm seed steps extra <<< "$spec"
  local armflag="--no-sr"; [ "$arm" = "sr" ] && armflag="--sr"
  local name="${algo}_${task}_${arm}_seed${seed}"
  local marker="$MARK/${name}.done"
  local log="$LOGS/${name}.log"

  if [ -f "$marker" ]; then
    echo "[skip ] $name"
    return 0
  fi
  echo "[start] $name (steps=$steps)"
  # shellcheck disable=SC2086
  if $PY scripts/train.py --algo "$algo" --task "$task" $armflag --seed "$seed" \
        --steps "$steps" --torch-threads "$THREADS" --logdir "$BASE" $extra \
        > "$log" 2>&1; then
    touch "$marker"
    echo "[done ] $name"
  else
    echo "[FAIL ] $name -- see $log (tail below)"
    tail -n 15 "$log" | sed 's/^/         /'
  fi
}

# --------------------------- concurrency loop ----------------------------- #
start_ts=$(date +%s)
for spec in "${JOBS[@]}"; do
  # Block until a slot frees up.
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_PAR" ]; do sleep 5; done
  run_one "$spec" &
  sleep 2   # stagger heavy init (env build, S2C alloc) so starts don't collide
done
wait

end_ts=$(date +%s)
done_n=$(find "$MARK" -name '*.done' | wc -l | tr -d ' ')
echo "=================================================================="
echo " ALL JOBS COMPLETE: $done_n / ${#JOBS[@]} markers present"
echo " wall-clock: $(( (end_ts - start_ts) / 3600 ))h $(( ((end_ts - start_ts) % 3600) / 60 ))m"
echo " next: python scripts/aggregate_results.py --base $BASE"
echo "=================================================================="
