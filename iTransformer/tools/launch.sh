#!/usr/bin/env bash
# Launch the iTransformer reproduction on the VM (RTX 4090).
# Usage: bash tools/launch.sh            -> start now (in background, nohup)
#        WAIT_PILOT=1 bash tools/launch.sh -> start, but block until the RG-Cycle pilot reports completion
set -euo pipefail
BASE=${BASE:-$HOME/PACFormer/repro_itransformer_20261119}
PY=${PY:-$HOME/PACFormer/.runtime/env/bin/python}
REPO=$BASE/repo/iTransformer
WORK=$BASE/work
DATA=$BASE/dataset
PILOT_STATUS=$HOME/PACFormer/rgcycle_20261119/pilot_day1_cyclenet/results/status.json
mkdir -p "$WORK"
EXTRA=()
if [[ "${WAIT_PILOT:-0}" == "1" ]]; then EXTRA+=(--wait-pilot-status "$PILOT_STATUS"); fi
export PYTHONPATH=$BASE/pydeps${PYTHONPATH:+:$PYTHONPATH}   # reformer_pytorch/einops (imported by official exp_basic.py)
nohup "$PY" -u "$REPO/tools/scheduler.py" \
  --jobs "$REPO/configs/jobs.csv" --data-root "$DATA" --work-dir "$WORK" --python "$PY" \
  --max-parallel 2 --ram-budget-gb 60 "${EXTRA[@]}" > "$WORK/scheduler.out" 2>&1 &
echo "scheduler pid $! ; log $WORK/scheduler.out ; status $WORK/results/status.json"
