"""Check B (pre-W1): are the two statistics redundant, and which one tracks
pair-level dependence (max lagged MI) better?

  G_ij   : cosine similarity of window descriptors [mean, log std]   (level similarity)
  rho_ij : max |lagged cross-correlation| over lags 0..MAXLAG        (lagged following)
  MI_ij  : max lagged mutual information over lags 0..MAXLAG         (pair dependence proxy)

Pure CPU, no training. Benchmark CSVs are read in place (read-only).
"""
import os
import sys
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_LOCAL = os.path.join(os.path.dirname(_HERE), 'data')
BASE = _LOCAL if os.path.isdir(_LOCAL) else r'C:\Users\Tim W\Desktop\PAC-Former\final\data\benchmark'
L = 96
MAXLAG = 12
N_WIN = 200
N_BINS = 8
SEED = 2026


def rankdata(x):
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1, dtype=np.float64)
    s = x[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            avg = (i + j) / 2.0 + 1.0
            ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks


def spearmanr(x, y):
    rx, ry = rankdata(x), rankdata(y)
    rx -= rx.mean()
    ry -= ry.mean()
    den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


def load_csv(path):
    for enc in ("utf-8-sig", "utf-8", "gbk", "latin-1"):
        try:
            df = pd.read_csv(path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    df = df.dropna(how="all")
    if "date" in df.columns:
        df = df.drop(columns=["date"])
    else:
        df = df.iloc[:, 1:]
    df = df.apply(pd.to_numeric, errors="coerce")
    arr = df.to_numpy(dtype=np.float64)
    return np.nan_to_num(arr)


def train_split(X):
    n_tr = int(len(X) * 0.5)
    tr = X[:n_tr]
    mu, sd = tr.mean(0), tr.std(0)
    sd = np.where(sd > 1e-8, sd, 1.0)
    return X[:n_tr].copy(), n_tr


def pair_stats(Xtr):
    C = Xtr.shape[1]
    Lw = min(L, len(Xtr))
    starts = np.unique(np.linspace(0, len(Xtr) - Lw, N_WIN).astype(int))
    G = np.zeros((C, C))
    RHO = np.zeros((C, C))
    for s in starts:
        w = Xtr[s:s + Lw]
        mu = w.mean(0)                      # per-channel window mean
        sd = w.std(0)                       # per-channel window std
        desc = np.stack([np.tanh(mu), np.tanh(np.log(sd + 1e-8))], 1)
        desc /= np.linalg.norm(desc, axis=1, keepdims=True) + 1e-12
        G += desc @ desc.T
        z = (w - mu) / (sd + 1e-8)
        z /= np.linalg.norm(z, axis=0, keepdims=True) + 1e-12
        for lag in range(0, MAXLAG + 1):
            if lag == 0:
                cc = z.T @ z
            else:
                cc = z[:-lag].T @ z[lag:]
            RHO = np.maximum(RHO, np.abs(cc))
    # RHO is accumulated once per (window, lag); normalise by both
    return G / len(starts), RHO / (len(starts) * (MAXLAG + 1))


def mi_lag(Xtr):
    C = Xtr.shape[1]
    n = len(Xtr)
    codes = np.zeros((n, C), dtype=np.int16)
    for c in range(C):
        q = np.quantile(Xtr[:, c], np.linspace(0, 1, N_BINS + 1)[1:-1])
        codes[:, c] = np.searchsorted(q, Xtr[:, c])
    bias = (N_BINS - 1) ** 2 / (2 * n)
    MI = np.zeros((C, C))
    for a in range(C):
        ca = codes[:, a]
        for b in range(C):
            if a == b:
                continue
            best = 0.0
            cb = codes[:, b]
            for lag in range(0, MAXLAG + 1):
                x = ca[: n - lag] if lag else ca
                y = cb[lag:] if lag else cb
                joint = np.zeros((N_BINS, N_BINS))
                np.add.at(joint, (x, y), 1)
                p = joint / joint.sum()
                px = p.sum(1, keepdims=True)
                py = p.sum(0, keepdims=True)
                with np.errstate(divide="ignore", invalid="ignore"):
                    mi = np.nansum(p * np.log(p / (px * py)))
                if mi - bias > best:
                    best = mi - bias
            MI[a, b] = best
    return MI


def offdiag(M):
    return M[~np.eye(M.shape[0], dtype=bool)]


def enrich(score, truth, top_q=0.9, good_q=0.75):
    k = max(2, int(np.ceil((1 - top_q) * len(score))))
    idx = np.argsort(-score)[:k]
    return float((truth[idx] >= np.quantile(truth, good_q)).mean() / (1 - good_q))


def run(files):
    rng = np.random.default_rng(SEED)
    rows = []
    for f in files:
        X = load_csv(os.path.join(BASE, f))
        Xtr, _ = train_split(X)
        C = Xtr.shape[1]
        keep = np.arange(C)
        if C > 40:
            keep = np.sort(rng.choice(C, 40, replace=False))
            Xtr = Xtr[:, keep]
        G, RHO = pair_stats(Xtr)
        MI = mi_lag(Xtr)
        g, r, m = offdiag(G), offdiag(RHO), offdiag(MI)
        row = dict(dataset=f.replace(".csv", ""), C=len(keep), pairs=len(m),
                   G_rho=round(spearmanr(g, r), 3),
                   G_MI=round(spearmanr(g, m), 3),
                   rho_MI=round(spearmanr(r, m), 3),
                   enrichG=round(enrich(g, m), 2),
                   enrichRho=round(enrich(r, m), 2),
                   rho_uniq=round(len(np.unique(np.round(r, 4))) / len(r), 2))
        rows.append(row)
        print(row, flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    files = sys.argv[1:] or ["ETTh1.csv", "ETTh2.csv", "ETTm1.csv", "ETTm2.csv", "Weather.csv"]
    df = run(files)
    print()
    print(df.to_string(index=False))
    OUT = os.path.join(os.path.dirname(_HERE), "results")
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, "checkB_results.csv"), index=False)

