"""v8: complete the fair table and start the switch from the other end.

Two groups, both on the shared CI encoder:
  plain        on the four ETT datasets (v7 compared mix against an OLD plain whose
               head had no window statistics -- not comparable)
  mix @ beta=1 on all six datasets: start at the dense-mixing end (which is what
               gains on Electricity) and let beta learn to turn OFF where mixing
               hurts. v7 started at 0.1 and beta never moved, so the switch never
               had a chance to switch on the dataset that needs it.
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
OUT = os.path.join(ROOT, "results/w12_v8.jsonl")
STATUS = os.path.join(ROOT, "STATUS_V8.md")

H, LR, BETA = 96, 3e-4, 1.0
SEEDS = [2026, 2027, 2028]
CFG = {
    "Electricity": (4, 32), "Weather": (2, 64),
    "ETTh1": (1, 64), "ETTh2": (1, 64), "ETTm1": (1, 64), "ETTm2": (1, 64),
}
ORDER = ["Electricity", "Weather", "ETTh1", "ETTh2", "ETTm1", "ETTm2"]
JOBS = [(d, "plain", 0.1) for d in ORDER] + [(d, "mix", BETA) for d in ORDER]


def done():
    s = set()
    if not os.path.exists(OUT):
        return s
    for line in open(OUT, encoding="utf-8"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        s.add((r["dataset"], r["arm"], round(float(r.get("beta_init", 0.1)), 3), r["seed"]))
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
    todo = [(d, a, b, s) for (d, a, b) in JOBS for s in SEEDS
            if (d, a, round(b, 3), s) not in have]
    print(f"todo={len(todo)}", flush=True)
    rows, fails = [], 0
    t0 = time.time()
    for i, (d, a, b, s) in enumerate(todo, 1):
        stride, batch = CFG[d]
        wait_idle(me)
        args = [PY, "train.py", "--dataset", d, "--arm", a, "--horizon", str(H),
                "--seed", str(s), "--out", OUT, "--stride", str(stride),
                "--batch", str(batch), "--epochs", "20", "--patience", "3",
                "--channels", "9999", "--lr", str(LR), "--d", "128", "--layers", "3",
                "--beta-init", str(b)]
        print(f"[{i}/{len(todo)}] {d} {a} beta={b} s{s}", flush=True)
        r = subprocess.run(args, cwd=ROOT)
        if r.returncode != 0:
            fails += 1
            print(f"  FAILED rc={r.returncode}", flush=True)
        else:
            try:
                rows.append(json.loads(open(OUT, encoding="utf-8").read().splitlines()[-1]))
            except Exception:
                pass
        lines = ["# v8", "", f"_updated {datetime.now():%H:%M:%S}_",
                 f"- {i-1}/{len(todo)}, fails={fails}", "",
                 "| dataset | arm | beta0 | seed | ep | test_mse | beta_mean |",
                 "|---|---|---|---|---|---|---|"]
        for r2 in sorted(rows, key=lambda x: (x["dataset"], x["arm"], x["seed"])):
            lines.append(f"| {r2['dataset']} | {r2['arm']} | {r2.get('beta_init','-')} | {r2['seed']} | "
                         f"{r2['best_epoch']} | {r2['test_mse']:.4f} | {r2.get('beta_mean','-')} |")
        open(STATUS, "w", encoding="utf-8").write("\n".join(lines))
    print(f"ALLDONE fails={fails}", flush=True)


if __name__ == "__main__":
    main()
