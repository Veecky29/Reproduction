"""Diagnostic: which rho definition actually tracks the MI "truth", and by how much
is any difference beyond small-sample noise?

Three definitions, all with true Pearson per lag (differ only in scope):
  full     whole training half, max over lags 0..MAXLAG          (one score per edge)
  window   per 96-step window -> max over lags -> mean over windows (the prior we ship)
  window_dt  same, computed on the detrended series              (the W1.1 fix)

Also prints checkB.py's original estimator, which is *not* a correlation (it divides a
running max by N_WIN*(MAXLAG+1) and gets values around 3e-4).

With C=7 there are only 42 ordered pairs, so a pair Spearman has SE ~ 0.16: the
bootstrap CI is printed so small-dataset differences are not over-read.
"""
import os
import sys

import numpy as np

import checkB as B
import checkB_deseason as D

NBOOT = 4000
SEED = 2026


def boot_ci(x, y, nboot=NBOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(x)
    vals = np.empty(nboot)
    for b in range(nboot):
        idx = rng.integers(0, n, n)
        vals[b] = B.spearmanr(x[idx], y[idx])
    vals = vals[np.isfinite(vals)]
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def rho_full(X):
    """Proper Pearson per lag over the whole series, max over lags (symmetric)."""
    n, C = X.shape
    R = np.zeros((C, C))
    for lag in range(D.MAXLAG + 1):
        A = X[: n - lag] if lag else X
        Bs = X[lag:] if lag else X
        A = A - A.mean(0)
        Bs = Bs - Bs.mean(0)
        na = np.linalg.norm(A, axis=0)
        nb = np.linalg.norm(Bs, axis=0)
        na[na < 1e-12] = 1.0
        nb[nb < 1e-12] = 1.0
        R = np.maximum(R, np.abs((A / na).T @ (Bs / nb)))
    np.fill_diagonal(R, 0.0)
    return R


def main(files):
    print(f"{'dataset':12s} {'pairs':>5s} | {'full':>6s} {'CI':>13s} | "
          f"{'window':>6s} {'CI':>13s} | {'win_dt':>6s} {'CI':>13s} | {'old':>6s}")
    for f in files:
        X, T = D.load_data(os.path.join(B.BASE, f))
        Xtr = X[: len(X) // 2]
        C = Xtr.shape[1]
        if C > 40:
            rng = np.random.default_rng(B.SEED)
            Xtr = Xtr[:, np.sort(rng.choice(C, 40, replace=False))]
        trend, seasonal, resid = D.classical_decompose(Xtr, T)
        Xdt = Xtr - trend
        _, R_old = B.pair_stats(Xtr)
        R_full = rho_full(Xtr)
        R_win = D.rho_windows(Xtr)
        R_win_dt = D.rho_windows(Xdt)
        m = B.offdiag(D.mi_lag(Xtr))
        cols, cells = [], []
        for name, R in (("full", R_full), ("window", R_win), ("win_dt", R_win_dt),
                        ("old", R_old)):
            r = B.offdiag(R)
            s = B.spearmanr(r, m)
            ci = boot_ci(r, m)
            cells.append(f"{s:6.3f} [{ci[0]:.2f},{ci[1]:.2f}]")
        print(f"{f.replace('.csv',''):12s} {len(m):5d} | {cells[0]} | {cells[1]} | "
              f"{cells[2]} | {B.spearmanr(B.offdiag(R_old), m):6.3f}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["ETTh1.csv", "ETTh2.csv", "ETTm1.csv", "ETTm2.csv",
                          "Weather.csv", "Electricity.csv"])
