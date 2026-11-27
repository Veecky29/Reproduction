#!/usr/bin/env bash
# Full Phase 0: prepare CSVs, train DLinear and iTransformer, then diagnose.
# This starts GPU training. Do not call it from CI.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "${ROOT}"

bash setup.sh
python prepare_data.py "$@"
python train_baselines.py "$@"
python diagnose.py "$@"
