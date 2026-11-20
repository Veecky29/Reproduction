#!/usr/bin/env bash
# Launch the PatchTST reproduction on the VM (RTX 4090).
# Do not start this while the DLinear (or any other) reproduction is still using the GPU.
# Usage: bash tools/launch.sh
set -euo pipefail
BASE=${BASE:-$HOME/PACFormer/repro_patchtst_20261120}
PY=${PY:-$HOME/PACFormer/.runtime/env/bin/python}
REPO=$BASE/repo/PatchTST
WORK=$BASE/work
DATA=${DATA:-$HOME/PACFormer/repro_itransformer_20261119/dataset}
mkdir -p "$WORK"
nohup "$PY" -u "$REPO/tools/scheduler.py" \
  --jobs "$REPO/configs/jobs.csv" --data-root "$DATA" --work-dir "$WORK" --python "$PY" \
  --max-parallel 2 --ram-budget-gb 60 > "$WORK/scheduler.out" 2>&1 &
echo "scheduler pid $! ; log $WORK/scheduler.out ; status $WORK/results/status.json"
