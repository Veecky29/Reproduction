"""Data pipeline for W1.2: benchmark CSVs -> windows + the rho edge prior.

Splits follow the standard Informer protocol:
  ETT*        12 / 4 / 4 months from the first timestamp
  Weather     70 / 10 / 20 by ratio
  Electricity 70 / 10 / 20 by ratio (optionally a channel subset, seed 2026,
              matching the subset used by SHEAF/scripts/checkB*.py)

The rho prior is computed on the training split only, on the trend+season residual,
as a single (C, C) edge score -- the W1.1b conclusion. Never touch val/test here.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

BENCH = {
    "ETTh1": dict(period=24, kind="ett"),
    "ETTh2": dict(period=24, kind="ett"),
    "ETTm1": dict(period=96, kind="ett"),
    "ETTm2": dict(period=96, kind="ett"),
    "Weather": dict(period=144, kind="ratio"),
    "Electricity": dict(period=24, kind="ratio"),
}


def read_dataset(root: str, name: str):
    df = pd.read_csv(os.path.join(root, f"{name}.csv"))
    dates = pd.to_datetime(df["date"])
    cols = [c for c in df.columns if c != "date"]
    X = df.drop(columns=["date"]).to_numpy(np.float32)
    return dates, X, cols


def split_bounds(name: str, dates: pd.Series) -> dict:
    """Official TSLib borders.

    ETT*: 12/4/4 *thirty-day* months, i.e. borders 8640 / 11520 / 14400 for hourly
    data (12*30*24, 16*30*24, 20*30*24) and the same multiples of 96 for 15-min
    data. NOTE: ETTh1 has 17420 rows, so the last ~3000 rows are never evaluated --
    using calendar months instead (8760/11712/end-of-series) silently swaps in a
    harder summer-2018 stretch and inflates every MSE by ~20%.
    Weather / Electricity: 70/10/20 by ratio.
    """
    n = len(dates)
    if BENCH[name]["kind"] == "ett":
        p = BENCH[name]["period"]
        b1, b2, b3 = 12 * 30 * p, 16 * 30 * p, 20 * 30 * p
        return dict(train=(0, b1), val=(b1, b2), test=(b2, min(b3, n)))
    a, b = int(n * 0.7), int(n * 0.8)
    return dict(train=(0, a), val=(a, b), test=(b, n))


def windows(X: np.ndarray, lo: int, hi: int, L: int, H: int, stride: int):
    """Sliding windows fully inside [lo, hi). Returns (Xw[N,L,C], Yw[N,H,C])."""
    lo = max(0, lo)
    hi = min(len(X), hi)
    starts = np.arange(lo, hi - L - H + 1, stride)
    if len(starts) == 0:
        raise ValueError(f"no windows in [{lo},{hi}) for L={L} H={H}")
    idx = starts[:, None] + np.arange(L)[None, :]
    tgt = starts[:, None] + L + np.arange(H)[None, :]
    return X[idx].astype(np.float32), X[tgt].astype(np.float32)


def build(name: str, root: str, L: int, H: int, stride_train: int = 1,
          channels: int | None = None, seed: int = 2026, drop=()):
    dates, X, cols = read_dataset(root, name)
    if drop:
        keep = [i for i, c in enumerate(cols) if c.strip() not in {d.strip() for d in drop}]
        if len(keep) != len(cols):
            X = X[:, keep]
    if channels is not None and X.shape[1] > channels:
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(X.shape[1], channels, replace=False))
        X = X[:, keep]
    sp = split_bounds(name, dates)

    # Global standardisation with training-split stats, BEFORE windowing: the
    # benchmark protocol. Without it MSE is dominated by the largest-magnitude
    # channel (ECL) and the arms become indistinguishable. RevIN still applies
    # per window on top of this.
    tr_lo, tr_hi = sp["train"]
    mu = X[tr_lo:tr_hi].mean(0, keepdims=True)
    sd = X[tr_lo:tr_hi].std(0, keepdims=True)
    sd[sd < 1e-8] = 1.0
    X = (X - mu) / sd

    out = {}
    for part in ("train", "val", "test"):
        lo, hi = sp[part]
        s = stride_train if part == "train" else 1
        out[part] = windows(X, lo, hi, L, H, s)
    prior = rho_prior(X[tr_lo:tr_hi], BENCH[name]["period"])
    return out, prior, X.shape[1]


# --------------------------------------------------------------- rho prior

def _ma_kernel(T: int) -> np.ndarray:
    if T % 2 == 0:
        w = np.ones(T + 1) / T
        w[0] = w[-1] = 1.0 / (2.0 * T)
    else:
        w = np.ones(T) / T
    return w


def _centered_ma(x: np.ndarray, T: int) -> np.ndarray:
    w = _ma_kernel(T)
    pad = min(len(w) // 2, len(x) - 1)
    xp = np.pad(x, pad, mode="reflect")
    return np.convolve(xp, w, mode="valid")[: len(x)]


def residual(X: np.ndarray, T: int) -> np.ndarray:
    """Additive trend+seasonal residual, per channel."""
    n, C = X.shape
    phase = np.arange(n) % T
    cnt = np.maximum(np.bincount(phase, minlength=T), 1).astype(np.float64)
    out = np.empty_like(X)
    for c in range(C):
        tr = _centered_ma(X[:, c].astype(np.float64), T)
        detr = X[:, c] - tr
        ph = np.bincount(phase, weights=detr, minlength=T) / cnt
        ph -= ph.mean()
        out[:, c] = X[:, c] - tr - ph[phase]
    return out


def rho_prior(Xtr: np.ndarray, period: int, n_win: int = 200, L: int = 96,
              maxlag: int = 12) -> np.ndarray:
    """(C, C) edge score: mean over windows of max_lag |Pearson| on the residual."""
    R = residual(Xtr, period)
    n, C = R.shape
    Lw = min(L, n)
    starts = np.unique(np.linspace(0, n - Lw, n_win).astype(int)) if n > Lw else np.array([0])
    out = np.zeros((C, C), dtype=np.float32)
    for s in starts:
        w = R[s:s + Lw]
        best = np.zeros((C, C))
        for lag in range(maxlag + 1):
            A = w[: Lw - lag] if lag else w
            B = w[lag:] if lag else w
            if len(A) < 4:
                continue
            A = A - A.mean(0)
            B = B - B.mean(0)
            na = np.linalg.norm(A, axis=0)
            nb = np.linalg.norm(B, axis=0)
            na[na < 1e-12] = 1.0
            nb[nb < 1e-12] = 1.0
            best = np.maximum(best, np.abs((A / na).T @ (B / nb)))
        out += best.astype(np.float32)
    out /= max(len(starts), 1)
    np.fill_diagonal(out, 0.0)
    return out
