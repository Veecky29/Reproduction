"""Pilot: do the window statistics that RevIN erases carry information of their own?

Four ways to use them, same CI encoder, same protocol (lr 3e-4, early stop, H=96,
3 seeds). `plain` (concatenated into the head) already ran in v8, so only the two
new arms are queued here.

  nostats  head(h)                                   -- RevIN affine only
  plain    head([h ; mu ; log sd])                   -- v8, reused
  film     h*(1+tanh(g)) + b, (g,b) = MLP(mu,log sd) -- conditioning

The prior question this must answer: is the ETTm gain (0.3518 -> 0.3418 on ETTm1,
0.1771 -> 0.1753 on ETTm2) reproducible under one run, and does it beat plain
RevIN affine clearly? Pre-registered stop rule: if the only difference is the
small ETTm effect, this line stops here.
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
OUT = os.path.join(ROOT, "results/w12_v9.jsonl")
STATUS = os.path.join(ROOT, "STATUS_V9.md")

H, LR = 96, 3e-4
SEEDS = [2026, 2027, 2028]
ARMS = ["nostats", "film"]
CFG = {
    "Electricity": (4, 32), "Weather": (2, 64),
    "ETTh1": (1, 64), "ETTh2": (1, 64), "ETTm1": (1, 64), "ETTm2": (1, 64),
}
ORDER = ["ETTm1", "ETTm2", "ETTh1", "ETTh2", "Weather", "Electricity"]


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
    todo = [(d, a, s) for d in ORDER for a in ARMS for s in SEEDS
            if (d, a, s) not in have]
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
        lines = ["# v9 stats pilot", "", f"_updated {datetime.now():%H:%M:%S}_",
                 f"- {i-1}/{len(todo)}, fails={fails}", "",
                 "| dataset | arm | seed | ep | test_mse |", "|---|---|---|---|---|"]
        for r2 in sorted(rows, key=lambda x: (x["dataset"], x["arm"], x["seed"])):
            lines.append(f"| {r2['dataset']} | {r2['arm']} | {r2['seed']} | {r2['best_epoch']} | "
                         f"{r2['test_mse']:.4f} |")
        open(STATUS, "w", encoding="utf-8").write("\n".join(lines))
    print(f"ALLDONE fails={fails}", flush=True)


if __name__ == "__main__":
    main()
