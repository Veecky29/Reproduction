"""Companion to checkB.py: top edges + tercile MRI + ASCII scatter for one dataset."""
import os
import sys
import numpy as np
import checkB as B


def ascii_scatter(x, y, width=48, height=16, xlabel="", ylabel="", title=""):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xb = np.linspace(np.quantile(x, 0.01), np.quantile(x, 0.99), width)
    yb = np.linspace(np.quantile(y, 0.01), np.quantile(y, 0.99), height)
    grid = np.zeros((height, width), dtype=int)
    for xi, yi in zip(x, y):
        ix = np.clip(np.searchsorted(xb, xi) - 1, 0, width - 1)
        iy = np.clip(np.searchsorted(yb, yi) - 1, 0, height - 1)
        grid[iy, ix] += 1
    chars = " .:-=+*#%@"
    print(title)
    for iy in range(height - 1, -1, -1):
        row = "".join(chars[min(len(chars) - 1, int(np.log1p(c) * 1.4))] for c in grid[iy])
        print(f"{yb[iy]:6.3f} |{row}|")
    print("       +" + "-" * width + "+")
    print(f"        {xlabel}  (cells = pair counts, log-scaled)")


def load(fname):
    X = B.load_csv(os.path.join(B.BASE, fname))
    Xtr, _ = B.train_split(X)
    C = Xtr.shape[1]
    if C > 40:
        rng = np.random.default_rng(B.SEED)
        keep = np.sort(rng.choice(C, 40, replace=False))
        Xtr = Xtr[:, keep]
    return Xtr


def detail(fname, topn=5):
    Xtr = load(fname)
    G, RHO = B.pair_stats(Xtr)
    MI = B.mi_lag(Xtr)
    C = Xtr.shape[1]
    iu = np.triu_indices(C, 1)
    pairs = [(i, j, G[i, j], RHO[i, j], MI[i, j]) for i, j in zip(*iu)]
    gi = np.argsort([-p[2] for p in pairs])
    ri = np.argsort([-p[3] for p in pairs])
    print(f"\n=== {fname}: top-{topn} by G (level similarity) ===")
    for k in gi[:topn]:
        i, j, g, r, m = pairs[k]
        print(f"  ({i:3d},{j:3d})  G={g:+.3f}  rho={r:.3e}  MI={m:.3f}")
    print(f"=== {fname}: top-{topn} by rho (lagged following) ===")
    for k in ri[:topn]:
        i, j, g, r, m = pairs[k]
        print(f"  ({i:3d},{j:3d})  G={g:+.3f}  rho={r:.3e}  MI={m:.3f}")
    Gv = np.array([p[2] for p in pairs])
    Rv = np.array([p[3] for p in pairs])
    Mv = np.array([p[4] for p in pairs])
    print(f"\n=== {fname}: median MI by tercile ===")
    for name, v in (("G", Gv), ("rho", Rv)):
        q1, q2 = np.quantile(v, 0.33), np.quantile(v, 0.66)
        lo, hi = Mv[v <= q1], Mv[v >= q2]
        print(f"  {name:3s}: low={np.median(lo):.4f}  high={np.median(hi):.4f}  ratio={np.median(hi)/(np.median(lo)+1e-12):.2f}x")
    return G, RHO, MI


if __name__ == "__main__":
    fname = sys.argv[1] if len(sys.argv) > 1 else "Weather.csv"
    G, RHO, MI = detail(fname)
    g, r, m = B.offdiag(G), B.offdiag(RHO), B.offdiag(MI)
    print(f"\nSpearman  G~rho={B.spearmanr(g, r):+.3f}   G~MI={B.spearmanr(g, m):+.3f}   rho~MI={B.spearmanr(r, m):+.3f}")
    ascii_scatter(g, r, xlabel="G (level similarity)", ylabel="rho", title=f"\n{fname}: G vs rho")
    ascii_scatter(r, m, xlabel="rho (lagged following)", ylabel="MI", title=f"\n{fname}: rho vs MI_lag")
    ascii_scatter(g, m, xlabel="G (level similarity)", ylabel="MI", title=f"\n{fname}: G vs MI_lag")
