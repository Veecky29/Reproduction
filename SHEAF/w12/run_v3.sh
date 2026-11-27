#!/usr/bin/env bash
set -u
PY="/root/miniconda3/bin/python"
OUT="results/w12_v3.jsonl"
for seed in 2026 2027 2028; do
  echo "=== Weather sheaf seed=$seed ==="
  $PY train.py --dataset Weather --arm sheaf --horizon 96 --seed $seed --out $OUT --d 128 --layers 3
done
for arm in plain concat; do
  echo "=== Weather $arm seed=2028 ==="
  if [ "$arm" = "concat" ]; then A="--d 144 --layers 3"; else A="--d 128 --layers 3"; fi
  $PY train.py --dataset Weather --arm $arm --horizon 96 --seed 2028 --out $OUT $A
done
echo ALLDONE
