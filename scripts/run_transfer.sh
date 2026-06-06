#!/usr/bin/env bash
# Figure 6 (transfer) launcher: train a source S2C on PointButton1, then run
# three PPO-Lag conditions on the PointGoal1 TARGET to test whether a frozen
# source safety representation transfers.
#
# Conditions (each 3 seeds, PPO-Lag on PointGoal1):
#   transfer : SR-PPO-Lag with the FROZEN PointButton1 S2C (zero-shot transfer)
#   random   : SR-PPO-Lag with a FROZEN RANDOM-init S2C   (control: shows the
#              *learned* representation matters, not just the extra input dims)
#   base     : vanilla PPO-Lag, no S2C                     (lower reference)
#
# The PointButton1 S2C has input dim 76; PointGoal1's 60-dim obs is zero-padded
# to 76 inside the wrapper (Option A; documented simplification of the paper's
# Appendix A.4 unified-LiDAR wrapper).
#
# Usage:
#   conda activate srpl && unset PYTHONPATH && cd ~/srpl-reproduction
#   bash scripts/run_transfer.sh
#   MAX_PAR=4 STEPS=1000000 bash scripts/run_transfer.sh
#
# Run this AFTER (or alongside, if the main batch has finished) the main grid.

set -uo pipefail

MAX_PAR="${MAX_PAR:-4}"
THREADS="${THREADS:-2}"
SEEDS="${SEEDS:-0 1 2}"
STEPS="${STEPS:-1000000}"          # PPO-Lag target horizon (match main PPO runs)
BASE="${BASE:-./experiments/transfer}"
PY="${PY:-python}"
S2C_CKPT="${S2C_CKPT:-$BASE/s2c_button1.pt}"
SRC_EPISODES="${SRC_EPISODES:-400}"
SRC_EPOCHS="${SRC_EPOCHS:-30}"
S2C_INPUT_DIM="${S2C_INPUT_DIM:-76}"   # PointButton1 raw obs dim

MARK="$BASE/_markers"; LOGS="$BASE/_logs"
mkdir -p "$MARK" "$LOGS"

echo "=================================================================="
echo " Fig 6 transfer | MAX_PAR=$MAX_PAR THREADS=$THREADS SEEDS='$SEEDS'"
echo " target steps=$STEPS | s2c_ckpt=$S2C_CKPT | s2c_input_dim=$S2C_INPUT_DIM"
echo "=================================================================="

# ---- Phase 1: train the source S2C on PointButton1 (once) ----
if [ -f "$S2C_CKPT" ]; then
  echo "[source] reusing existing S2C checkpoint: $S2C_CKPT"
else
  echo "[source] training source S2C on PointButton1 ..."
  $PY scripts/train_source_s2c.py --episodes "$SRC_EPISODES" \
      --epochs "$SRC_EPOCHS" --out "$S2C_CKPT" 2>&1 | tee "$LOGS/source_s2c.log"
  if [ ! -f "$S2C_CKPT" ]; then
    echo "[source] FAILED to produce $S2C_CKPT -- aborting."; exit 1
  fi
fi

# ---- Phase 2: three target conditions on PointGoal1 ----
# Each entry: "condition|extra-args"
JOBS=()
for s in $SEEDS; do
  JOBS+=("transfer|$s|--sr --frozen-s2c --load-s2c $S2C_CKPT --s2c-input-dim $S2C_INPUT_DIM")
  JOBS+=("random|$s|--sr --frozen-s2c --s2c-input-dim $S2C_INPUT_DIM")
  JOBS+=("base|$s|--no-sr")
done

run_one() {
  local spec="$1"
  local cond seed extra
  IFS='|' read -r cond seed extra <<< "$spec"
  local name="${cond}_seed${seed}"
  local marker="$MARK/${name}.done"
  local log="$LOGS/${name}.log"
  local logdir="$BASE/$cond"
  if [ -f "$marker" ]; then echo "[skip ] $name"; return 0; fi
  echo "[start] $name"
  # shellcheck disable=SC2086
  if $PY scripts/train.py --algo PPOLag --task PointGoal1 --seed "$seed" \
        --steps "$STEPS" --torch-threads "$THREADS" --logdir "$logdir" $extra \
        > "$log" 2>&1; then
    touch "$marker"; echo "[done ] $name"
  else
    echo "[FAIL ] $name -- see $log"; tail -n 15 "$log" | sed 's/^/         /'
  fi
}

for spec in "${JOBS[@]}"; do
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_PAR" ]; do sleep 5; done
  run_one "$spec" &
  sleep 2
done
wait
echo "=================================================================="
echo " transfer runs complete. next:"
echo "   python scripts/plot_transfer.py --base $BASE"
echo "=================================================================="
