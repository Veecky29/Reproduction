# reproduction

Faithful reproductions of time-series forecasting baselines with their official code and
hyperparameters, 3 seeds each (2026, 2027, 2028). Each model has its own folder with code,
configs and results.

| Model | Folder | Status |
|---|---|---|
| iTransformer (ICLR 2024) | [iTransformer/](iTransformer/) | completed 108/108, 0 failed; wall clock ~17 h on RTX 4090, 2026-11-19 10:24 to 2026-11-20 03:22 MYT |
| DLinear (AAAI 2023) | [DLinear/](DLinear/) | running |

## iTransformer: results

Lookback 96. MSE / MAE on the normalised test set, mean ± sample std over seeds 2026/2027/2028.
"Paper" = Table 10 of arXiv:2310.06625v4. ΔMSE vs paper = (ours − paper) / paper.
Protocol and hyperparameters: [iTransformer/README.md](iTransformer/README.md).
A longer write-up is in [iTransformer/results/README.md](iTransformer/results/README.md).

<!-- RESULTS:BEGIN -->
| Dataset | Horizon | Seeds | MSE (mean ± std) | MAE (mean ± std) | Paper MSE | Paper MAE | ΔMSE vs paper | ΔMAE vs paper |
|---|---|---|---|---|---|---|---|---|
| ETTh1 | 96 | 3 | 0.3875 ± 0.0003 | 0.4051 ± 0.0003 | 0.386 | 0.405 | +0.38% | +0.03% |
| ETTh1 | 192 | 3 | 0.4425 ± 0.0003 | 0.4372 ± 0.0002 | 0.441 | 0.436 | +0.34% | +0.28% |
| ETTh1 | 336 | 3 | 0.4917 ± 0.0049 | 0.4615 ± 0.0036 | 0.487 | 0.458 | +0.97% | +0.77% |
| ETTh1 | 720 | 3 | 0.5112 ± 0.0142 | 0.4952 ± 0.0085 | 0.503 | 0.491 | +1.62% | +0.85% |
| ETTh2 | 96 | 3 | 0.3008 ± 0.0025 | 0.3506 ± 0.0015 | 0.297 | 0.349 | +1.27% | +0.45% |
| ETTh2 | 192 | 3 | 0.3794 ± 0.0002 | 0.3987 ± 0.0005 | 0.380 | 0.400 | -0.15% | -0.31% |
| ETTh2 | 336 | 3 | 0.4217 ± 0.0013 | 0.4318 ± 0.0011 | 0.428 | 0.432 | -1.48% | -0.04% |
| ETTh2 | 720 | 3 | 0.4385 ± 0.0071 | 0.4519 ± 0.0041 | 0.427 | 0.445 | +2.69% | +1.55% |
| ETTm1 | 96 | 3 | 0.3430 ± 0.0026 | 0.3772 ± 0.0016 | 0.334 | 0.368 | +2.69% | +2.51% |
| ETTm1 | 192 | 3 | 0.3824 ± 0.0021 | 0.3951 ± 0.0008 | 0.377 | 0.391 | +1.42% | +1.05% |
| ETTm1 | 336 | 3 | 0.4201 ± 0.0014 | 0.4190 ± 0.0007 | 0.426 | 0.420 | -1.39% | -0.23% |
| ETTm1 | 720 | 3 | 0.4914 ± 0.0026 | 0.4582 ± 0.0015 | 0.491 | 0.459 | +0.08% | -0.17% |
| ETTm2 | 96 | 3 | 0.1837 ± 0.0013 | 0.2681 ± 0.0029 | 0.180 | 0.264 | +2.07% | +1.55% |
| ETTm2 | 192 | 3 | 0.2513 ± 0.0014 | 0.3121 ± 0.0012 | 0.250 | 0.309 | +0.50% | +1.00% |
| ETTm2 | 336 | 3 | 0.3151 ± 0.0008 | 0.3523 ± 0.0012 | 0.311 | 0.348 | +1.33% | +1.24% |
| ETTm2 | 720 | 3 | 0.4110 ± 0.0025 | 0.4058 ± 0.0021 | 0.412 | 0.407 | -0.24% | -0.30% |
| Exchange | 96 | 3 | 0.0872 ± 0.0007 | 0.2075 ± 0.0006 | 0.086 | 0.206 | +1.44% | +0.73% |
| Exchange | 192 | 3 | 0.1794 ± 0.0007 | 0.3023 ± 0.0007 | 0.177 | 0.299 | +1.35% | +1.10% |
| Exchange | 336 | 3 | 0.3354 ± 0.0021 | 0.4203 ± 0.0014 | 0.331 | 0.417 | +1.32% | +0.80% |
| Exchange | 720 | 3 | 0.8484 ± 0.0053 | 0.6951 ± 0.0022 | 0.847 | 0.691 | +0.16% | +0.60% |
| Weather | 96 | 3 | 0.1748 ± 0.0010 | 0.2145 ± 0.0016 | 0.174 | 0.214 | +0.43% | +0.23% |
| Weather | 192 | 3 | 0.2247 ± 0.0005 | 0.2575 ± 0.0003 | 0.221 | 0.254 | +1.66% | +1.38% |
| Weather | 336 | 3 | 0.2809 ± 0.0018 | 0.2985 ± 0.0014 | 0.278 | 0.296 | +1.04% | +0.84% |
| Weather | 720 | 3 | 0.3588 ± 0.0025 | 0.3503 ± 0.0017 | 0.358 | 0.347 | +0.23% | +0.94% |
| ECL | 96 | 3 | 0.1481 ± 0.0003 | 0.2398 ± 0.0003 | 0.148 | 0.240 | +0.08% | -0.08% |
| ECL | 192 | 3 | 0.1647 ± 0.0009 | 0.2556 ± 0.0008 | 0.162 | 0.253 | +1.64% | +1.04% |
| ECL | 336 | 3 | 0.1792 ± 0.0009 | 0.2724 ± 0.0011 | 0.178 | 0.269 | +0.68% | +1.27% |
| ECL | 720 | 3 | 0.2119 ± 0.0046 | 0.3007 ± 0.0027 | 0.225 | 0.317 | -5.80% | -5.14% |
| Solar-Energy | 96 | 3 | 0.2063 ± 0.0003 | 0.2364 ± 0.0017 | 0.203 | 0.237 | +1.61% | -0.26% |
| Solar-Energy | 192 | 3 | 0.2394 ± 0.0004 | 0.2629 ± 0.0005 | 0.233 | 0.261 | +2.73% | +0.72% |
| Solar-Energy | 336 | 3 | 0.2499 ± 0.0010 | 0.2742 ± 0.0007 | 0.248 | 0.273 | +0.75% | +0.45% |
| Solar-Energy | 720 | 3 | 0.2503 ± 0.0007 | 0.2759 ± 0.0006 | 0.249 | 0.275 | +0.52% | +0.34% |
| Traffic | 96 | 3 | 0.3933 ± 0.0004 | 0.2690 ± 0.0004 | 0.395 | 0.268 | -0.42% | +0.37% |
| Traffic | 192 | 3 | 0.4128 ± 0.0002 | 0.2773 ± 0.0003 | 0.417 | 0.276 | -1.00% | +0.48% |
| Traffic | 336 | 3 | 0.4233 ± 0.0009 | 0.2825 ± 0.0001 | 0.433 | 0.283 | -2.24% | -0.16% |
| Traffic | 720 | 3 | 0.4593 ± 0.0013 | 0.3008 ± 0.0008 | 0.467 | 0.302 | -1.64% | -0.41% |
<!-- RESULTS:END -->

## DLinear: results

Lookback 96. MSE / MAE on the standardised test set, mean ± sample std over seeds 2026/2027/2028.
"Paper" = PAC-Former Table 4 = iTransformer Table 10 (arXiv:2310.06625v4). Official L=96 numbers
from LTSF-Benchmark.md are also shown where they exist. Protocol: [DLinear/README.md](DLinear/README.md).

<!-- DLINEAR_RESULTS:BEGIN -->
*Pending: the 96 runs (8 datasets × 4 horizons × 3 seeds) have started on the RTX 4090.*
<!-- DLINEAR_RESULTS:END -->
