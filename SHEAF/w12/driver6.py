"""v7: switchable residual mixing.

Arms:
  mix     h'_i = h_i + beta_i * Mix(h)_i , beta init 0.1 (beta=0 -> plain, beta=1 -> concat)
  concat  dense free mixing on the same encoder (already measured for ECL/Weather in
          v6; re-run here for the four ETT datasets, whose old concat used a
          different encoder and is not comparable)

Runs: mix on all six datasets x 3 seeds, plus concat on the four ETT datasets x 3
seeds. Same encoder, same protocol (lr 3e-4, early stop patience 3).
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime

ROOT = "/root/autodl-tmp/sheaf_w12"
DATA = "/root/autodl-tmp/benchmark"
PY = "/root/miniconda3/bin/python"
OUT = os.path.join(ROOT, "results/w12_v7.jsonl")
STATUS = os.path.join(ROOT, "STATUS_V7.md")

H, LR = 96, 3e-4
SEEDS = [2026, 2027, 2028]
CFG = {
    "Electricity": (4, 32), "Weather": (2, 64),
    "ETTh1": (1, 64), "ETTh2": (1, 64), "ETTm1": (1, 64), "ETTm2": (1, 64),
}
ORDER = ["Electricity", "Weather", "ETTh1", "ETTh2", "ETTm1", "ETTm2"]
# (dataset, arm): ECL/Weather concat already exist in v6
PAIRS = [(d, "mix") for d in ORDER] + [(d, "concat") for d in ("ETTh1", "ETTh2", "ETTm1", "ETTm2")]


def done():
    s = set()
    if not os.path.exists(OUT):
        return s
    for line in open(OUT, encoding="utf-8"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        s.add((r["dataset"], r["arm"], r["seed"]))
    return s


def wait_idle(me):
    while True:
        o = subprocess.run(["pgrep", "-f", "train.py"], capture_output=True, text=True)
        pids = [p for p in o.stdout.split() if p.isdigit() and int(p) != me]
        if not pids:
            return
        time.sleep(10)


def main():
    me = os.getpid()
    have = done()
    todo = [(d, a, s) for (d, a) in PAIRS for s in SEEDS if (d, a, s) not in have]
    print(f"todo={len(todo)}", flush=True)
    rows, fails = [], 0
    t0 = time.time()
    for i, (d, a, s) in enumerate(todo, 1):
        stride, batch = CFG[d]
        wait_idle(me)
        args = [PY, "train.py", "--dataset", d, "--arm", a, "--horizon", str(H),
                "--seed", str(s), "--out", OUT, "--stride", str(stride),
                "--batch", str(batch), "--epochs", "20", "--patience", "3",
                "--channels", "9999", "--lr", str(LR), "--d", "128", "--layers", "3"]
        print(f"[{i}/{len(todo)}] {d} {a} s{s}", flush=True)
        r = subprocess.run(args, cwd=ROOT)
        if r.returncode != 0:
            fails += 1
            print(f"  FAILED rc={r.returncode}", flush=True)
        else:
            try:
                rows.append(json.loads(open(OUT, encoding="utf-8").read().splitlines()[-1]))
            except Exception:
                pass
        lines = ["# v7 switchable mixing", "", f"_updated {datetime.now():%H:%M:%S}_",
                 f"- {i-1}/{len(todo)}, fails={fails}", "",
                 "| dataset | arm | seed | ep | test_mse | beta_mean |", "|---|---|---|---|---|---|"]
        for r2 in sorted(rows, key=lambda x: (x["dataset"], x["arm"], x["seed"])):
            lines.append(f"| {r2['dataset']} | {r2['arm']} | {r2['seed']} | {r2['best_epoch']} | "
                         f"{r2['test_mse']:.4f} | {r2.get('beta_mean','-')} |")
        open(STATUS, "w", encoding="utf-8").write("\n".join(lines))
    print(f"ALLDONE fails={fails}", flush=True)


if __name__ == "__main__":
    main()
