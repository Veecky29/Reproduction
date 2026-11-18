# reproduction

Faithful reproductions of time-series forecasting baselines with their official code and
hyperparameters, 3 seeds each (2026, 2027, 2028). Each model has its own folder with code,
configs and results.

| Model | Folder | Status |
|---|---|---|
| iTransformer (ICLR 2024) | [`iTransformer/`](iTransformer/) | code and configs ready, runs pending |

## iTransformer: results

Lookback 96. MSE / MAE on the normalised test set, mean ± sample std over seeds 2026/2027/2028.
"Paper" = Table 10 of arXiv:2310.06625v4. ΔMSE vs paper = (ours − paper) / paper.
Protocol and hyperparameters: [`iTransformer/README.md`](iTransformer/README.md).

<!-- RESULTS:BEGIN -->
*Pending: the 108 runs (9 datasets × 4 horizons × 3 seeds) have not started yet.*
<!-- RESULTS:END -->
