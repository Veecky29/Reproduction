#!/usr/bin/env bash
# Launch the FEDformer reproduction on the VM (RTX 4090).
# Usage: bash tools/launch.sh
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
DATA=${DATA:-/root/autodl-tmp/fed_data}
WORK=${WORK:-/root/autodl-tmp/fed_work}
PY=${PY:-/root/miniconda3/bin/python}
mkdir -p "$WORK"
nohup "$PY" -u "$REPO/tools/scheduler.py" \
  --jobs "$REPO/configs/jobs.csv" --data-root "$DATA" --work-dir "$WORK" --python "$PY" \
  --max-parallel 1 --ram-budget-gb 60 > "$WORK/scheduler.out" 2>&1 &
echo "scheduler pid $! ; log $WORK/scheduler.out"
