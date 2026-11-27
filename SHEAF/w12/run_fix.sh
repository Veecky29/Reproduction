#!/usr/bin/env bash
set -u
PY="/root/miniconda3/bin/python"
OUT="results/w12_fix.jsonl"
mkdir -p results
for ds in ETTh1 Weather; do
  for seed in 2026 2027; do
    for arm in plain concat sheaf; do
      case "$arm" in
        concat) A="--d 144 --layers 3" ;;
        *)      A="--d 128 --layers 3" ;;
      esac
      echo "=== $ds H=96 $arm seed=$seed ==="
      $PY train.py --dataset "$ds" --arm "$arm" --horizon 96 --seed "$seed" --out "$OUT" $A
    done
  done
done
echo ALLDONE
