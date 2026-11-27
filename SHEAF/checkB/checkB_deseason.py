"""W1.1: does the edge prior rho survive removal of shared seasonality / trend?

The confound under test: rho_ij and MI_ij are both computed on raw series, where a
common daily cycle makes every pair co-move. If rho only tracks "both channels ride
the same clock", the edge prior is degenerate (all edges saturate) and useless for
attribution. This script re-estimates BOTH sides on the same de-confounded signal.

Two fixes relative to checkB.py:
  1. rho is now a true Pearson coefficient: per lag the two truncated segments are
     demeaned separately and normalised by their own norms; per window we take
     max_lag |cc| and average over windows (checkB mixed a running max over
     windows*and* lags, then divided by a constant, so its values were not rho).
  2. MI is recomputed in the same domain as rho for every variant, so the Spearman
     is apples-to-apples. The original "raw vs raw" number is kept as a reference.

Variants (all per channel, fitted on the training half only):
  raw            x
  detrend_only   x - centered moving average of length T   (shared slow trend gone, season kept)
  deseason_only  x - classical seasonal component           (shared cycle gone, trend kept)
  resid          x - trend - season                          (both gone)
  season_diff    x[t] - x[t-T]                               (robustness, no decomposition assumptions)

Pure CPU, no training. Writes results/checkB_deseason.csv; never overwrites checkB_results.csv.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

import checkB as B

L = 96
MAXLAG = 12
N_WIN = 200
N_BINS = 8
SEED = 2026

VARIANTS = ("raw", "detrend_only", "deseason_only", "resid", "season_diff")


# --------------------------------------------------------------------------- io

def load_data(path):
    """Return (X[n, C], period_in_steps_per_day)."""
    for enc in ("utf-8-sig", "utf-8", "gbk", "latin-1"):
        try:
            df = pd.read_csv(path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    df = df.dropna(how="all")
    if "date" not in df.columns:
        df = df.rename(columns={df.columns[0]: "date"})
    dates = pd.to_datetime(df["date"], errors="coerce", format="mixed")
    num = df.drop(columns=["date"]).apply(pd.to_numeric, errors="coerce")
    X = np.nan_to_num(num.to_numpy(dtype=np.float64))
    return X, infer_period(dates)


def infer_period(dates):
    d = dates.dropna().diff().dt.total_seconds().dropna()
    d = d[d > 0]
    if len(d) == 0:
        return 24
    step = float(np.median(d.to_numpy()))
    T = int(round(86400.0 / step))
    return int(np.clip(T, 2, 2000))


# ------------------------------------------------------------ preprocessing

def _ma_kernel(T):
    """Centered moving-average kernel; even T gets the usual half-weight endpoints."""
    if T % 2 == 0:
        w = np.ones(T + 1, dtype=np.float64) / T
        w[0] = w[-1] = 1.0 / (2.0 * T)
    else:
        w = np.ones(T, dtype=np.float64) / T
    return w


def centered_ma(x, T):
    """Centered MA along axis 0 of a 1-D or 2-D array, reflect-padded at the edges."""
    w = _ma_kernel(T)
    pad = len(w) // 2
    n = x.shape[0]
    pad = min(pad, n - 1)
    xp = np.pad(x, ((pad, pad), (0, 0))) if x.ndim == 2 else np.pad(x, pad, mode="reflect")
    y = np.apply_along_axis(lambda c: np.convolve(c, w, mode="valid"), 0, xp)
    return y[:n]


def classical_decompose(X, T):
    """Additive trend+seasonal decomposition, per channel. Returns (trend, seasonal, resid)."""
    n, C = X.shape
    trend = np.empty_like(X)
    seasonal = np.empty_like(X)
    phase = np.arange(n) % T
    cnt = np.bincount(phase, minlength=T).astype(np.float64)
    for c in range(C):
        tr = centered_ma(X[:, c], T)
        detr = X[:, c] - tr
        ph = np.bincount(phase, weights=detr, minlength=T) / np.maximum(cnt, 1.0)
        ph -= ph.mean()
        trend[:, c] = tr
        seasonal[:, c] = ph[phase]
    return trend, seasonal, X - trend - seasonal


def make_variants(Xtr, T):
    trend, seasonal, resid = classical_decompose(Xtr, T)
    return {
        "raw": Xtr,
        "detrend_only": Xtr - trend,
        "deseason_only": Xtr - seasonal,
        "resid": resid,
        "season_diff": Xtr[T:] - Xtr[:-T],
    }


# ---------------------------------------------------------------- statistics

def rho_windows(X):
    """Mean over windows of max_{lag<=MAXLAG} |Pearson(i, j at that lag)|."""
    n, C = X.shape
    Lw = min(L, n)
    if Lw < 8:
        return np.zeros((C, C))
    starts = np.unique(np.linspace(0, n - Lw, N_WIN).astype(int))
    R = np.zeros((C, C))
    for s in starts:
        w = X[s:s + Lw]
        best = np.zeros((C, C))
        for lag in range(MAXLAG + 1):
            A = w[: Lw - lag] if lag else w
            Bs = w[lag:] if lag else w
            if len(A) < 4:
                continue
            A = A - A.mean(0)
            Bs = Bs - Bs.mean(0)
            na = np.linalg.norm(A, axis=0)
            nb = np.linalg.norm(Bs, axis=0)
            na[na < 1e-12] = 1.0
            nb[nb < 1e-12] = 1.0
            cc = (A / na).T @ (Bs / nb)
            best = np.maximum(best, np.abs(cc))
        R += best
    R /= len(starts)
    np.fill_diagonal(R, 0.0)
    return R


def mi_lag(X):
    """max_{lag} bias-corrected mutual information, 8 quantile bins (checkB semantics)."""
    n, C = X.shape
    codes = np.zeros((n, C), dtype=np.int64)
    qs = np.linspace(0, 1, N_BINS + 1)[1:-1]
    for c in range(C):
        codes[:, c] = np.searchsorted(np.quantile(X[:, c], qs), X[:, c])
    bias = (N_BINS - 1) ** 2 / (2.0 * n)
    MI = np.zeros((C, C))
    # MI[a, b] = max_lag MI(a[t], b[t+lag]), i.e. "b leads a" is a[b] not a[a] here:
    # the entry is directed, exactly as in checkB.mi_lag, so the off-diagonal vector
    # holds both directions of every pair (2 * C*(C-1)/2 entries).
    for a in range(C):
        ca = codes[:, a]
        for b in range(C):
            if a == b:
                continue
            cb = codes[:, b]
            best = 0.0
            for lag in range(MAXLAG + 1):
                x = ca[: n - lag] if lag else ca
                y = cb[lag:] if lag else cb
                joint = np.bincount(x * N_BINS + y, minlength=N_BINS * N_BINS)
                p = joint.reshape(N_BINS, N_BINS).astype(np.float64)
                p /= p.sum()
                px = p.sum(1, keepdims=True)
                py = p.sum(0, keepdims=True)
                with np.errstate(divide="ignore", invalid="ignore"):
                    mi = np.nansum(p * np.log(p / (px * py)))
                if mi - bias > best:
                    best = mi - bias
            MI[a, b] = best
    return MI


def enrich(score, truth, top_q=0.9, good_q=0.75):
    k = max(2, int(np.ceil((1 - top_q) * len(score))))
    idx = np.argsort(-score)[:k]
    return float((truth[idx] >= np.quantile(truth, good_q)).mean() / (1 - good_q))


# -------------------------------------------------------------------- driver

def analyse(Xtr, T, label):
    out = {}
    for name, V in make_variants(Xtr, T).items():
        V = np.nan_to_num(V, nan=0.0, posinf=0.0, neginf=0.0)
        RHO = rho_windows(V)
        MI = mi_lag(V)
        r, m = B.offdiag(RHO), B.offdiag(MI)
        q = np.quantile(r, [0.10, 0.50, 0.90])
        out[name] = dict(
            RHO=RHO,
            row=dict(dataset=label, variant=name, C=Xtr.shape[1], pairs=len(m),
                     rho_MI=round(B.spearmanr(r, m), 3),
                     enrichRho=round(enrich(r, m), 2),
                     rho_p10=round(float(q[0]), 3),
                     rho_p50=round(float(q[1]), 3),
                     rho_p90=round(float(q[2]), 3),
                     rho_sat=round(float((r >= 0.9).mean()), 2),
                     mi_mean=round(float(m.mean()), 4)),
        )
    base = out["raw"]["RHO"]
    for name in VARIANTS[1:]:
        r, b = B.offdiag(out[name]["RHO"]), B.offdiag(base)
        out[name]["row"]["rho_vs_raw"] = round(B.spearmanr(r, b), 3)
    out["raw"]["row"]["rho_vs_raw"] = 1.0
    return out


def main(files):
    rows = []
    for f in files:
        path = os.path.join(B.BASE, f)
        X, T = load_data(path)
        Xtr = X[: len(X) // 2]
        C = Xtr.shape[1]
        if C > 40:
            rng = np.random.default_rng(SEED)
            keep = np.sort(rng.choice(C, 40, replace=False))
            Xtr = Xtr[:, keep]
        label = f.replace(".csv", "")
        out = analyse(Xtr, T, label)
        for name in VARIANTS:
            rows.append(out[name]["row"])
        print(f"[{label}] C={Xtr.shape[1]} T={T} "
              + "  ".join(f"{n}:{out[n]['row']['rho_MI']:.3f}" for n in VARIANTS),
              flush=True)
    df = pd.DataFrame(rows)
    OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, "checkB_deseason.csv"), index=False)

    print("\n=== Spearman(rho, MI) by variant ===")
    print(df.pivot(index="dataset", columns="variant", values="rho_MI")[list(VARIANTS)].to_string())
    print("\n=== rho p90 (saturation check) ===")
    print(df.pivot(index="dataset", columns="variant", values="rho_p90")[list(VARIANTS)].to_string())
    print("\n=== fraction of edges with rho >= 0.9 ===")
    print(df.pivot(index="dataset", columns="variant", values="rho_sat")[list(VARIANTS)].to_string())
    print("\n=== rho rank agreement with raw ===")
    print(df.pivot(index="dataset", columns="variant", values="rho_vs_raw")[list(VARIANTS)].to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", default=None)
    a = ap.parse_args()
    main(a.files or ["ETTh1.csv", "ETTh2.csv", "ETTm1.csv", "ETTm2.csv",
                     "Weather.csv", "Electricity.csv"])
