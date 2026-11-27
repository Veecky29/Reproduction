"""Check B, part 2: full-training-series version and partial correlations.

  rho_full : one max-|lagged corr| computed on the whole training series (not averaged over windows)
  partial  : does G add anything beyond rho (and vice versa), controlling for the other?
"""
import os
import numpy as np
import pandas as pd
import checkB as B


def rho_full(Xtr, maxlag=12):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    z = (Xtr - mu) / sd
    z /= np.linalg.norm(z, axis=0, keepdims=True) + 1e-12
    C = z.shape[1]
    R = np.zeros((C, C))
    for lag in range(0, maxlag + 1):
        cc = z[:-lag].T @ z[lag:] if lag else z.T @ z
        R = np.maximum(R, np.abs(cc))
    return R


def partial_spearman(x, y, z):
    """Spearman of x,y controlling for z (rank-residual method)."""
    rx, ry, rz = B.rankdata(x), B.rankdata(y), B.rankdata(z)
    A = np.c_[np.ones_like(rz), rz]
    bx = np.linalg.lstsq(A, rx, rcond=None)[0]
    by = np.linalg.lstsq(A, ry, rcond=None)[0]
    ex, ey = rx - A @ bx, ry - A @ by
    den = np.sqrt((ex ** 2).sum() * (ey ** 2).sum())
    return float((ex * ey).sum() / den)


def main(files):
    rows = []
    for f in files:
        X = B.load_csv(os.path.join(B.BASE, f))
        Xtr, _ = B.train_split(X)
        C = Xtr.shape[1]
        if C > 40:
            rng = np.random.default_rng(B.SEED)
            keep = np.sort(rng.choice(C, 40, replace=False))
            Xtr = Xtr[:, keep]
        G, RHO_win = B.pair_stats(Xtr)
        RHO = rho_full(Xtr)
        MI = B.mi_lag(Xtr)
        g, rw, r, m = B.offdiag(G), B.offdiag(RHO_win), B.offdiag(RHO), B.offdiag(MI)
        rows.append(dict(
            dataset=f.replace(".csv", ""), C=Xtr.shape[1], pairs=len(m),
            G_rho=round(B.spearmanr(g, r), 3),
            G_MI=round(B.spearmanr(g, m), 3),
            rho_MI=round(B.spearmanr(r, m), 3),
            partial_G_given_rho=round(partial_spearman(g, m, r), 3),
            partial_rho_given_G=round(partial_spearman(r, m, g), 3),
            win_vs_full=round(B.spearmanr(rw, r), 3),
        ))
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    print()
    print(df.to_string(index=False))
    OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, "checkB_partial.csv"), index=False)


if __name__ == "__main__":
    main(["ETTh1.csv", "ETTh2.csv", "ETTm1.csv", "ETTm2.csv", "Weather.csv"])

