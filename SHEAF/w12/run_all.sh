#!/usr/bin/env bash
# W1.2 matrix runner.  usage: bash run_all.sh [quick]
#
# Capacity matching: concat is run at d=144 so its parameter count (790k) lands
# within 3% of sheaf (770k) instead of 18% below it. plain and sheaf share d=128.
# Everything else (lr, batch, epochs, patience, seed) is identical across arms.
set -u

MODE="${1:-full}"
OUT="results/w12.jsonl"
mkdir -p results
PY="${PY:-/root/miniconda3/bin/python}"
command -v "$PY" >/dev/null || { echo "interpreter not found: $PY"; exit 1; }

if [ "$MODE" = "quick" ]; then
  DATASETS="ETTh1"
  HORIZONS="96"
  SEEDS="2026 2027"
else
  DATASETS="ETTh1 ETTh2 ETTm1 ETTm2 Weather Electricity"
  HORIZONS="96 192 336"
  SEEDS="2026 2027"
fi

arm_args() {
  case "$1" in
    concat) echo "--d 144 --layers 3" ;;
    *)      echo "--d 128 --layers 3" ;;
  esac
}

for ds in $DATASETS; do
  for H in $HORIZONS; do
    for arm in plain concat sheaf; do
      for seed in $SEEDS; do
        echo "=== $ds H=$H $arm seed=$seed ==="
        $PY train.py --dataset "$ds" --arm "$arm" --horizon "$H" \
                     --seed "$seed" --out "$OUT" $(arm_args "$arm") \
          || echo "FAILED: $ds $H $arm $seed"
      done
    done
  done
done

echo "done -> $OUT"
wc -l "$OUT"
