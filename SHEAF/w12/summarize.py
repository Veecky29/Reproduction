"""Aggregate results/w12.jsonl into the W1.2 decision table.

    python summarize.py results/w12.jsonl

Prints per (dataset, horizon): mean+-std test MSE for the three arms, the paired
delta of `sheaf` against `concat`, and the pre-registered verdict.
"""
from __future__ import annotations

import collections
import json
import sys

import numpy as np

TIE = 0.01      # |relative delta| below this counts as a tie
WIN = 0.02      # sheaf must beat concat by at least this to count as a win


def load(path):
    rows = []
    with open(path, encoding="utf-8-sig") as f:   # tolerate a BOM from Windows tools
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def verdict(delta, consistent):
    """delta = (sheaf - concat)/concat, so negative means sheaf has lower MSE."""
    if abs(delta) < TIE or not consistent:
        return "TIE"
    if delta <= -WIN:
        return "WIN"
    if delta >= TIE:
        return "LOSS"
    return "marginal"


def main(path):
    rows = load(path)
    if not rows:
        print("no records")
        return
    g = collections.defaultdict(list)
    for r in rows:
        g[(r["dataset"], r["horizon"], r["arm"])].append(r)

    keys = sorted({(d, h) for d, h, _ in g}, key=lambda t: (t[0], t[1]))
    arms = ["plain", "concat", "sheaf"]
    print(f"{'dataset':12s} {'H':>4s} | " +
          " | ".join(f"{a:>14s}" for a in arms) + f" | {'sheaf vs concat':>16s} | params")
    print("-" * 110)

    verdicts = []
    for ds, H in keys:
        cells, means, params = [], {}, set()
        for a in arms:
            recs = g.get((ds, H, a), [])
            if not recs:
                cells.append(f"{'--':>14s}")
                continue
            m = np.array([r["test_mse"] for r in recs])
            means[a] = m.mean()
            params.update(r["params"] for r in recs)
            cells.append(f"{m.mean():9.4f}卤{m.std():.4f}")

        # paired delta: compare seeds present in both arms
        d = "--"
        v = "n/a"
        if "concat" in means and "sheaf" in means:
            by_seed = {a: {r["seed"]: r["test_mse"] for r in g[(ds, H, a)]} for a in ("sheaf", "concat")}
            common = sorted(set(by_seed["sheaf"]) & set(by_seed["concat"]))
            if common:
                ds_delta = [(by_seed["sheaf"][s] - by_seed["concat"][s]) / by_seed["concat"][s]
                            for s in common]
                delta = float(np.mean(ds_delta))
                consistent = all(x < 0 for x in ds_delta) or all(x > 0 for x in ds_delta)
                d = f"{100 * delta:+.2f}%"
                v = verdict(delta, consistent)
                verdicts.append((ds, H, v))
        print(f"{ds:12s} {H:4d} | " + " | ".join(cells) +
              f" | {d:>16s} | {sorted(params)}")

    if verdicts:
        n = collections.Counter(v for _, _, v in verdicts)
        print("\nverdict:", dict(n))
        print("  TIE  -> trivial baseline matches the readout: cross-channel structure adds nothing")
        print("  WIN  -> readout earns its keep, proceed to W2")
        print("  LOSS -> readout hurts, revisit the design before spending more GPU hours")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/w12.jsonl")
