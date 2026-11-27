"""Corrected full matrix: lr=3e-4, early stopping (the literature protocol), 3 seeds.

Two things this fixes versus the overnight run:
  * lr 3e-4 (the per-dataset search picked it on all six datasets; 1e-3 was used
    before and cost up to 13% on Electricity);
  * ConcatMix channel embedding sized to C, so Electricity (321) and Traffic (862)
    no longer crash.

Order puts Electricity and Weather first: Electricity is the only benchmark where
channel mixing is worth ~25% (iTransformer 0.148 vs DLinear 0.197) and Weather is
where PAC-Former's own ablation shows the mechanism working. If sheaf loses to
concat on both, the project is done and we stop.

Idempotent.  nohup python driver4.py > logs/driver4.log 2>&1 &
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
OUT = os.path.join(ROOT, "results/w12_v5.jsonl")
STATUS = os.path.join(ROOT, "STATUS_V5.md")

H, LR = 96, 3e-4
SEEDS = [2026, 2027, 2028]
ARMS = ["plain", "concat", "sheaf"]

# order matters: decisive datasets first
ORDER = ["Electricity", "Weather", "ETTh1", "ETTh2", "ETTm1", "ETTm2", "Traffic"]
CFG = {  # stride, batch
    "Electricity": (4, 32), "Weather": (2, 64),
    "ETTh1": (1, 64), "ETTh2": (1, 64), "ETTm1": (1, 64), "ETTm2": (1, 64),
    "Traffic": (8, 32),
}


def avail():
    return [d for d in ORDER if os.path.exists(os.path.join(DATA, f"{d}.csv"))]


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
    todo = [(d, a, s) for d in avail() for a in ARMS for s in SEEDS
            if (d, a, s) not in have]
    print(f"available={avail()}  todo={len(todo)}", flush=True)
    rows, fails = [], 0
    t0 = time.time()
    for i, (d, a, s) in enumerate(todo, 1):
        stride, batch = CFG[d]
        wait_idle(me)
        args = [PY, "train.py", "--dataset", d, "--arm", a, "--horizon", str(H),
                "--seed", str(s), "--out", OUT, "--stride", str(stride),
                "--batch", str(batch), "--epochs", "20", "--patience", "3",
                "--channels", "9999", "--lr", str(LR)]
        args += ["--d", "144", "--layers", "3"] if a == "concat" else ["--d", "128", "--layers", "3"]
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
        lines = [f"# v5 matrix (lr={LR}, early stop)", "",
                 f"_updated {datetime.now():%Y-%m-%d %H:%M:%S}_",
                 f"- {i-1}/{len(todo)} done, {fails} failures"]
        if i > 1:
            per = (time.time() - t0) / (i - 1)
            lines.append(f"- ETA {(len(todo)-i)*per/3600:.1f} h")
        lines += ["", "| dataset | arm | seed | ep | test_mse |", "|---|---|---|---|---|"]
        for r2 in sorted(rows[-50:], key=lambda x: (x["dataset"], x["arm"], x["seed"])):
            lines.append(f"| {r2['dataset']} | {r2['arm']} | {r2['seed']} | "
                         f"{r2['best_epoch']} | {r2['test_mse']:.4f} |")
        open(STATUS, "w", encoding="utf-8").write("\n".join(lines))
    print(f"ALLDONE fails={fails}", flush=True)


if __name__ == "__main__":
    main()
