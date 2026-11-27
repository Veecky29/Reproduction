#!/usr/bin/env bash
set -u
PY="/root/miniconda3/bin/python"
OUT="results/w12_v4.jsonl"
for seed in 2026 2027 2028; do
  for arm in plain concat sheaf; do
    if [ "$arm" = "concat" ]; then A="--d 144 --layers 3"; else A="--d 128 --layers 3"; fi
    echo "=== Weather $arm seed=$seed ==="
    $PY train.py --dataset Weather --arm $arm --horizon 96 --seed $seed \
        --out $OUT --fixed-epochs --epochs 10 $A
  done
done
echo ALLDONE
