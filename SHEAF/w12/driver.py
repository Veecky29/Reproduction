"""Autonomous W1.2 matrix driver.  Idempotent: safe to kill and restart.

Scans results/*.jsonl for finished (dataset, horizon, arm, seed) tuples, then runs
only the missing ones, appending to results/matrix.jsonl and rewriting STATUS.md
after every run.  Waits for any other train.py to finish before starting a run, so
it can be launched while another job is still going.

    nohup python driver.py > logs/driver.log 2>&1 &
    kill $(pgrep -f driver.py)     # stop after the current run
"""
from __future__ import annotations

import glob
import itertools
import json
import os
import subprocess
import time
from datetime import datetime

ROOT = "/root/autodl-tmp/sheaf_w12"
PY = "/root/miniconda3/bin/python"
OUT = os.path.join(ROOT, "results/matrix.jsonl")
STATUS = os.path.join(ROOT, "STATUS.md")

# Full W1.2 matrix. Weather gets all three horizons (that is where the signal is);
# the four ETT datasets stay at H=96 because they are saturated and 3x slower.
DATASETS = ["Weather"]
HORIZONS = [96, 192, 336]
SEEDS = [2026, 2027, 2028]
ARMS = ["plain", "concat", "sheaf"]

ETT_DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]
ETT_HORIZONS = [96]
ETT_SEEDS = [2026, 2027, 2028]


def targets():
    t = list(itertools.product(DATASETS, HORIZONS, ARMS, SEEDS))
    t += list(itertools.product(ETT_DATASETS, ETT_HORIZONS, ARMS, ETT_SEEDS))
    return t


def finished_keys():
    seen = set()
    for p in glob.glob(os.path.join(ROOT, "results", "*.jsonl")):
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            seen.add((r["dataset"], r["horizon"], r["arm"], r["seed"]))
    return seen


def wait_idle(exclude_pid):
    """Block until no other train.py process is running (not counting ourselves)."""
    while True:
        out = subprocess.run(["pgrep", "-f", "train.py"], capture_output=True, text=True)
        pids = [p for p in out.stdout.split() if p.isdigit() and int(p) != exclude_pid]
        if not pids:
            return
        time.sleep(10)


def write_status(rows, done_n, total_n, elapsed, current):
    done = rows
    lines = [f"# STATUS — W1.2 matrix", "", f"_updated {datetime.now():%Y-%m-%d %H:%M:%S}_", ""]
    lines.append(f"- progress: **{done_n}/{total_n}** configs")
    if current:
        lines.append(f"- running now: `{current}`")
    if done_n and elapsed > 0:
        per = elapsed / done_n
        lines.append(f"- avg {per/60:.1f} min/config, "
                     f"ETA {(total_n - done_n) * per / 3600:.1f} h")
    lines += ["", "| dataset | H | arm | seed | best_ep | test_mse |", "|---|---|---|---|---|---|"]
    for r in sorted(done, key=lambda x: (x["dataset"], x["horizon"], x["arm"], x["seed"]))[-60:]:
        lines.append(f"| {r['dataset']} | {r['horizon']} | {r['arm']} | {r['seed']} | "
                     f"{r['best_epoch']} | {r['test_mse']:.4f} |")
    lines += ["", "Stop: `kill $(pgrep -f driver.py)` (current run finishes first).", ""]
    with open(STATUS, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    me = os.getpid()
    want = targets()
    have = finished_keys()
    todo = [t for t in want if t not in have]
    print(f"{len(want)} target configs, {len(have & set(want))} already done, "
          f"{len(todo)} to run", flush=True)

    rows = []
    if os.path.exists(OUT):
        for line in open(OUT, encoding="utf-8"):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    t0 = time.time()
    for i, (ds, h, arm, seed) in enumerate(todo, 1):
        wait_idle(me)
        cur = f"{ds} H={h} {arm} s{seed}"
        write_status(rows, i - 1, len(todo), time.time() - t0, cur)
        args = [PY, "train.py", "--dataset", ds, "--arm", arm, "--horizon", str(h),
                "--seed", str(seed), "--out", OUT]
        args += ["--d", "144", "--layers", "3"] if arm == "concat" else ["--d", "128", "--layers", "3"]
        print(f"[{i}/{len(todo)}] {cur}", flush=True)
        subprocess.run(args, cwd=ROOT)
    write_status(rows, len(todo), len(todo), time.time() - t0, None)
    print("ALLDONE", flush=True)


if __name__ == "__main__":
    main()
