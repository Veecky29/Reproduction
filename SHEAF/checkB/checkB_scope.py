"""2x2 control: does the rho->MI agreement come from scope (full vs window) or from
domain (raw vs de-confounded), and are the two even comparable?

The truth metric MI is itself global and level-sensitive, so a global rho looks better
against it by construction. Only domain-matched cells are meaningful:

  domain=raw    rho and MI both on the series the model sees
  domain=resid  rho and MI both on trend+season residuals (what a per-channel model
                cannot explain from that channel's own history)

Last column: paired bootstrap (over edges) of window-minus-full Spearman in the resid
domain, i.e. whether the per-window choice is justified by this criterion at all.
"""
import os

import numpy as np

import checkB as B
import checkB_deseason as D
from checkB_estimator import rho_full

FILES = ["ETTh1.csv", "ETTh2.csv", "ETTm1.csv", "ETTm2.csv",
         "Weather.csv", "Electricity.csv"]
NBOOT = 2000


def main():
    print(f"{'dataset':12s} {'edges':>5s} | {'full/raw':>8s} {'win/raw':>7s} | "
          f"{'full/resid':>10s} {'win/resid':>9s} | {'win-full [95% CI]':>19s}")
    for f in FILES:
        X, T = D.load_data(os.path.join(B.BASE, f))
        Xtr = X[: len(X) // 2]
        C = Xtr.shape[1]
        if C > 40:
            rng = np.random.default_rng(B.SEED)
            Xtr = Xtr[:, np.sort(rng.choice(C, 40, replace=False))]
        _, _, resid = D.classical_decompose(Xtr, T)
        resid = np.nan_to_num(resid)
        cells = {}
        for dom, S in (("raw", Xtr), ("resid", resid)):
            m = B.offdiag(D.mi_lag(S))
            cells[(dom, "full")] = B.spearmanr(B.offdiag(rho_full(S)), m)
            cells[(dom, "win")] = B.spearmanr(B.offdiag(D.rho_windows(S)), m)
        m = B.offdiag(D.mi_lag(resid))
        rw, rf = B.offdiag(D.rho_windows(resid)), B.offdiag(rho_full(resid))
        rng = np.random.default_rng(B.SEED)
        ds = np.empty(NBOOT)
        for k in range(NBOOT):
            idx = rng.integers(0, len(m), len(m))
            ds[k] = B.spearmanr(rw[idx], m[idx]) - B.spearmanr(rf[idx], m[idx])
        lo, hi = np.quantile(ds, [0.025, 0.975])
        print(f"{f.replace('.csv',''):12s} {len(m):5d} | "
              f"{cells[('raw','full')]:8.3f} {cells[('raw','win')]:7.3f} | "
              f"{cells[('resid','full')]:10.3f} {cells[('resid','win')]:9.3f} | "
              f"{cells[('resid','win')] - cells[('resid','full')]:+7.3f} "
              f"[{lo:+.3f},{hi:+.3f}]", flush=True)


if __name__ == "__main__":
    main()
