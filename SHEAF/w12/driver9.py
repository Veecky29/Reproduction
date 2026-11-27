"""Decisive check: is FiLM's -11.9% on Weather just a patch for the near-degenerate
rain channels?

Weather's `rain (mm)` / `raining (s)` are zero in most windows (window std median 0),
and a first version of this pipeline blew up on them (RevIN floor 1e-5 -> 1e8 target
variance). If FiLM's gain comes from acting as a learned per-channel scale gate that
suppresses those channels, dropping them should remove most of the gain.

  variant  'all'   all 21 channels
           'norain' the two rain columns removed (19 channels)

Arms: nostats and film, 3 seeds each. Same protocol.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime

ROOT = "/root/autodl-tmp/sheaf_w12"
PY = "/root/miniconda3/bin/python"
OUT = os.path.join(ROOT, "results/w12_v10.jsonl")
STATUS = os.path.join(ROOT, "STATUS_V10.md")

H, LR = 96, 3e-4
SEEDS = [2026, 2027, 2028]
ARMS = ["nostats", "film"]
VARIANTS = [("all", ""), ("norain", "rain (mm),raining (s)")]


def done():
    s = set()
    if not os.path.exists(OUT):
        return s
    for line in open(OUT, encoding="utf-8"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        s.add((r.get("variant"), r["arm"], r["seed"]))
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
    todo = [(v, drop, a, s) for (v, drop) in VARIANTS for a in ARMS for s in SEEDS
            if (v, a, s) not in have]
    print(f"todo={len(todo)}", flush=True)
    rows, fails = [], 0
    for i, (v, drop, a, s) in enumerate(todo, 1):
        wait_idle(me)
        args = [PY, "train.py", "--dataset", "Weather", "--arm", a, "--horizon", str(H),
                "--seed", str(s), "--out", OUT, "--stride", "2", "--batch", "64",
                "--epochs", "20", "--patience", "3", "--channels", "9999",
                "--lr", str(LR), "--d", "128", "--layers", "3", "--drop-channels", drop]
        print(f"[{i}/{len(todo)}] {v} {a} s{s}", flush=True)
        r = subprocess.run(args, cwd=ROOT)
        if r.returncode != 0:
            fails += 1
            print(f"  FAILED rc={r.returncode}", flush=True)
        else:
            line = open(OUT, encoding="utf-8").read().splitlines()[-1]
            rec = json.loads(line)
            rec["variant"] = v
            with open(OUT, "a", encoding="utf-8") as f:
                pass
            rows.append(rec)
    # rewrite OUT with the variant tag so downstream code can split
    if os.path.exists(OUT):
        fixed = []
        for line in open(OUT, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "variant" not in r:
                r["variant"] = "all"
            fixed.append(r)
        with open(OUT, "w", encoding="utf-8") as f:
            for r in fixed:
                f.write(json.dumps(r) + "\n")
    print(f"ALLDONE fails={fails}", flush=True)


if __name__ == "__main__":
    main()
