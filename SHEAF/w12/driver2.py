"""Full-benchmark W1.2 driver (v2).  Idempotent, failure-tolerant, unattended.

Runs every (dataset, arm, seed) that is not already in results/*.jsonl, with
per-dataset window stride / batch size. Fixed-epoch protocol (no early stopping)
because val bottoms out at epoch 1-3 on these benchmarks and best-val selection
adds ~15% seed variance.

    nohup python driver2.py > logs/driver2.log 2>&1 &

Progress: STATUS.md.  Results: results/w12_full.jsonl.
A dataset whose CSV is missing is skipped, so traffic can be added later by just
dropping Traffic.csv into the data root and restarting the driver.
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import time
from datetime import datetime

ROOT = "/root/autodl-tmp/sheaf_w12"
DATA = "/root/autodl-tmp/benchmark"
PY = "/root/miniconda3/bin/python"
OUT = os.path.join(ROOT, "results/w12_full.jsonl")
STATUS = os.path.join(ROOT, "STATUS.md")

H = 96
SEEDS = [2026, 2027, 2028]
ARMS = ["plain", "concat", "sheaf"]

# dataset -> (train stride, batch, epochs)
CFG = {
    "ETTh1": (1, 64, 10),
    "ETTh2": (1, 64, 10),
    "ETTm1": (1, 64, 10),
    "ETTm2": (1, 64, 10),
    "Weather": (2, 64, 10),
    "Electricity": (4, 32, 10),
    "Traffic": (8, 32, 10),
}


def avail():
    return [d for d in CFG if os.path.exists(os.path.join(DATA, f"{d}.csv"))]


# Only runs produced by the CURRENT model (target-conditioned readout) and the
# CURRENT protocol (fixed epochs) may count as "done". Older files used the
# simplified readout / best-val selection and must be re-run.
CURRENT = [OUT, os.path.join(ROOT, "results/w12_v4.jsonl")]


def finished():
    seen = set()
    for p in CURRENT:
        if not os.path.exists(p):
            continue
        fh = open(p, encoding="utf-8")
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("horizon") == H:
                seen.add((r["dataset"], r["arm"], r["seed"]))
    return seen


def wait_idle(me):
    while True:
        out = subprocess.run(["pgrep", "-f", "train.py"], capture_output=True, text=True)
        pids = [p for p in out.stdout.split() if p.isdigit() and int(p) != me]
        if not pids:
            return
        time.sleep(10)


def write_status(rows, i, n, t0, cur, fails):
    lines = [f"# STATUS 鈥?full-benchmark W1.2", "",
             f"_updated {datetime.now():%Y-%m-%d %H:%M:%S}_", "",
             f"- progress: **{i}/{n}**", f"- running: `{cur}`" if cur else "- idle"]
    if i:
        per = (time.time() - t0) / i
        lines.append(f"- avg {per / 60:.1f} min/config, ETA {(n - i) * per / 3600:.1f} h")
    if fails:
        lines.append(f"- failures: {fails}")
    lines += ["", "| dataset | arm | seed | ep | test_mse |", "|---|---|---|---|---|"]
    for r in sorted(rows[-40:], key=lambda x: (x["dataset"], x["arm"], x["seed"])):
        lines.append(f"| {r['dataset']} | {r['arm']} | {r['seed']} | "
                     f"{r['best_epoch']} | {r['test_mse']:.4f} |")
    lines += ["", "Stop: `kill $(pgrep -f driver2.py)`"]
    with open(STATUS, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    me = os.getpid()
    todo = [(d, a, s) for d in avail() for a in ARMS for s in SEEDS
            if (d, a, s) not in finished()]
    print(f"available: {avail()}", flush=True)
    print(f"{len(todo)} configs to run", flush=True)
    rows, fails = [], 0
    t0 = time.time()
    for i, (d, a, s) in enumerate(todo, 1):
        stride, batch, epochs = CFG[d]
        wait_idle(me)
        cur = f"{d} {a} s{s}"
        write_status(rows, i - 1, len(todo), t0, cur, fails)
        args = [PY, "train.py", "--dataset", d, "--arm", a, "--horizon", str(H),
                "--seed", str(s), "--out", OUT, "--stride", str(stride),
                "--batch", str(batch), "--epochs", str(epochs), "--fixed-epochs",
                "--channels", "9999"]
        args += ["--d", "144", "--layers", "3"] if a == "concat" else ["--d", "128", "--layers", "3"]
        print(f"[{i}/{len(todo)}] {cur}", flush=True)
        r = subprocess.run(args, cwd=ROOT)
        if r.returncode != 0:
            fails += 1
            print(f"  FAILED rc={r.returncode}: {cur}", flush=True)
        else:
            for line in open(OUT, encoding="utf-8").read().splitlines()[-1:]:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    write_status(rows, len(todo), len(todo), t0, None, fails)
    print(f"ALLDONE fails={fails}", flush=True)


if __name__ == "__main__":
    main()

