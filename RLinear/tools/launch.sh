#!/usr/bin/env bash
# Launch the RLinear reproduction on the VM (RTX 4090).
# Do not start this while PatchTST (or any other) reproduction is still using the GPU.
# Usage:
#   bash tools/launch.sh
#   MAX_PARALLEL=2 NUM_WORKERS=10 RAM_BUDGET_GB=60 bash tools/launch.sh
# Official num_workers is 10. When sharing the GPU with another job, use NUM_WORKERS=2
# and consider MAX_PARALLEL=1 so dataloader workers do not slow the other run.
set -euo pipefail
BASE=${BASE:-$HOME/PACFormer/repro_rlinear_20261121}
PY=${PY:-$HOME/PACFormer/.runtime/env/bin/python}
REPO=$BASE/repo/RLinear
WORK=$BASE/work
DATA=${DATA:-$HOME/PACFormer/repro_itransformer_20261119/dataset}
MAX_PARALLEL=${MAX_PARALLEL:-2}
RAM_BUDGET_GB=${RAM_BUDGET_GB:-60}
NUM_WORKERS=${NUM_WORKERS:-10}
mkdir -p "$WORK"
nohup "$PY" -u "$REPO/tools/scheduler.py" \
  --jobs "$REPO/configs/jobs.csv" --data-root "$DATA" --work-dir "$WORK" --python "$PY" \
  --max-parallel "$MAX_PARALLEL" --ram-budget-gb "$RAM_BUDGET_GB" --num-workers "$NUM_WORKERS" \
  > "$WORK/scheduler.out" 2>&1 &
echo "scheduler pid $! ; log $WORK/scheduler.out ; status $WORK/results/status.json"
