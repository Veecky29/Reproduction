"""Per-dataset learning-rate search on the `plain` arm only.

Protocol: tune the backbone once per dataset, then freeze that lr for all three
arms. Tuning on `plain` (which uses no cross-channel information) means the search
cannot favour `sheaf` -- exactly the point.

Fixed 10 epochs, no early stopping, same as the main matrix.
Results append to results/tune.jsonl; STATUS_TUNE.md is rewritten each run.
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
OUT = os.path.join(ROOT, "results/tune.jsonl")
STATUS = os.path.join(ROOT, "STATUS_TUNE.md")

H, SEED = 96, 2026
LRS = [3e-4, 1e-3, 3e-3]

CFG = {
    "ETTh1": (1, 64), "ETTh2": (1, 64), "ETTm1": (1, 64), "ETTm2": (1, 64),
    "Weather": (2, 64), "Electricity": (4, 32), "Traffic": (8, 32),
}


def done():
    s = set()
    if not os.path.exists(OUT):
        return s
    for line in open(OUT, encoding="utf-8"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        s.add((r["dataset"], r["lr"]))
    return s


def main():
    have = done()
    avail = [d for d in CFG if os.path.exists(os.path.join(DATA, f"{d}.csv"))]
    todo = [(d, lr) for d in avail for lr in LRS if (d, lr) not in have]
    print(f"tune todo {len(todo)} over {avail}", flush=True)
    t0 = time.time()
    for i, (d, lr) in enumerate(todo, 1):
        stride, batch = CFG[d]
        print(f"[{i}/{len(todo)}] {d} lr={lr}", flush=True)
        args = [PY, "train.py", "--dataset", d, "--arm", "plain", "--horizon", str(H),
                "--seed", str(SEED), "--out", OUT, "--stride", str(stride),
                "--batch", str(batch), "--epochs", "10", "--fixed-epochs",
                "--channels", "9999", "--d", "128", "--layers", "3", "--lr", str(lr)]
        subprocess.run(args, cwd=ROOT)
        lines = [f"# LR search (plain arm)", "", f"_updated {datetime.now():%H:%M:%S}_", ""]
        if os.path.exists(OUT):
            rows = [json.loads(x) for x in open(OUT, encoding="utf-8") if x.strip()]
            for r in rows:
                lines.append(f"| {r['dataset']} | {r['lr']} | {r['test_mse']:.4f} |")
        with open(STATUS, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    print("TUNEDONE", flush=True)


if __name__ == "__main__":
    main()
