# DLinear reproduction

Faithful re-run of **DLinear** (Zeng et al., AAAI 2023, arXiv:2205.13504; official
code: [cure-lab/LTSF-Linear](https://github.com/cure-lab/LTSF-Linear)) with the
authors' official code and official script hyperparameters, 3 seeds (2026, 2027, 2028).

This is the next baseline in the PAC-Former comparison. PAC-Former Table 4 copies
DLinear's numbers from iTransformer (Liu et al., ICLR 2024) Appendix F.4 Table 10 at
**lookback L = 96**, not from the DLinear paper's default lookback of 336.

## What is reproduced

* Long-term multivariate forecasting, lookback **96**, horizons 96 / 192 / 336 / 720.
  Datasets: ETTh1, ETTh2, ETTm1, ETTm2, Exchange, Weather, ECL (Electricity), Traffic.
  8 datasets x 4 horizons x 3 seeds = **96 runs** (`configs/jobs.csv`).
* PAC-Former Table 4 uses 7 of these (no Exchange). Exchange is included because it
  is in iTransformer Table 10 and has an official DLinear script.
* **Solar-Energy is not in the default job list.** Official LTSF-Linear has no Solar
  script, and `Dataset_Custom` only reads a CSV with a `date` column, so `solar_AL.txt`
  cannot be loaded without changing official code. Table 10 numbers are recorded in
  `configs/paper_reference.csv` for reference only.
* PEMS and ILI are not part of this run.

## Protocol

* Code: `official/` = cure-lab/LTSF-Linear @ `0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6`.
  Only change: `--seed` argument in `run_longExp.py` (see `patches/official_changes.diff`).
  Official default seed is 2021.
* Hyperparameters: taken verbatim from `official/scripts/EXP-LongForecasting/Linear/*.sh`
  except **`seq_len` is set to 96** (the official scripts hard-code 336). Anything a
  script does not set falls back to the `run_longExp.py` default. `tools/build_configs.py`
  generates `configs/jobs.csv` from those files (the `*_src` columns say where each
  value came from). No hyperparameter search. `LOOKBACK=336 python tools/build_configs.py`
  will emit a verbatim official-lookback job list if that protocol is chosen later.
* Model: DLinear (shared linear layers), **not** DLinear-I. Official scripts do not
  pass `--individual`. Decomposition kernel size is hard-coded at 25.
* Fixed by the official code: Adam, MSE loss, `train_epochs` 10, early stopping on
  validation MSE with patience 3, `lradj=type1` (lr halves every epoch). The test
  loader uses the training `batch_size` with `drop_last=False`, so every test window
  is evaluated. Metrics are computed on the standardised scale. DLinear does not use
  time features; `--freq` is left at the official default.
* Seeds: 2026, 2027, 2028 (`--seed`); official scripts use `--itr 1` and hard-code 2021.
* Main numbers are **mean ± sample std over the 3 seeds**, never the best seed.
* Primary reference values (`paper_mse` / `paper_mae`) are PAC-Former Table 4 =
  iTransformer Table 10 (L=96). `official_l96_*` columns (ECL, Exchange, Traffic,
  Weather only) are the DLinear authors' own L=96 numbers from `LTSF-Benchmark.md`.
  Those two sources already disagree, especially on Traffic and ECL (see below).
  ETT official numbers in that file use "best input length" (typically 336) and are
  stored as `official_bestL_*`, not as L=96 references.

## Lookback 96 vs 336

The official DLinear scripts and the DLinear paper's main ETT results use lookback
336; the README states that LTSF-Linear "generally achieves better results with a
longer look-back window". PAC-Former and iTransformer Table 10 fix L=96 for every
baseline, and PAC-Former §4.5 says it uses those fixed-L96 values "rather than
substituting results obtained with a different or optimized lookback".

This reproduction follows L=96, official script learning rates and batch sizes,
because the goal is the baseline PAC-Former actually compared against. The ETT
"best lookback" numbers in `LTSF-Benchmark.md` should not be used as the L=96
target.

## Hyperparameters (official scripts, seq_len overridden to 96; seed-independent)

| Dataset | Horizon | enc_in | batch | lr | epochs | patience | lradj | individual | Official script | lr / batch source |
|---|---|---|---|---|---|---|---|---|---|---|
| ETTh1 | 96 / 192 / 336 / 720 | 7 | 32 | 0.005 | 10 | 3 | type1 | 0 | `Linear/etth1.sh` | script |
| ETTh2 | 96 / 192 / 336 / 720 | 7 | 32 | 0.05 | 10 | 3 | type1 | 0 | `Linear/etth2.sh` | script |
| Exchange | 96, 192 | 8 | 8 | 0.0005 | 10 | 3 | type1 | 0 | `Linear/exchange_rate.sh` | script |
| Exchange | 336, 720 | 8 | 32 | 0.0005 | 10 | 3 | type1 | 0 | `Linear/exchange_rate.sh` | script |
| ETTm1 | 96 / 192 / 336 / 720 | 7 | 8 | 0.0001 | 10 | 3 | type1 | 0 | `Linear/ettm1.sh` | script |
| ETTm2 | 96, 192 | 7 | 32 | 0.001 | 10 | 3 | type1 | 0 | `Linear/ettm2.sh` | script |
| ETTm2 | 336, 720 | 7 | 32 | 0.01 | 10 | 3 | type1 | 0 | `Linear/ettm2.sh` | script |
| Weather | 96 / 192 / 336 / 720 | 21 | 16 | 0.0001 | 10 | 3 | type1 | 0 | `Linear/weather.sh` | batch from script; **lr is the `run_longExp.py` default** (script does not set it) |
| ECL | 96 / 192 / 336 / 720 | 321 | 16 | 0.001 | 10 | 3 | type1 | 0 | `Linear/electricity.sh` | script |
| Traffic | 96 / 192 / 336 / 720 | 862 | 16 | 0.05 | 10 | 3 | type1 | 0 | `Linear/traffic.sh` | script |

## Paper numbers vs official L=96 numbers

iTransformer says its baselines were "implemented based on the benchmark of TimesNet
... fairly built on the configurations provided by each model's original paper or
official code" at lookback 96. Current Time-Series-Library DLinear scripts exist
only for ETTh1 and ECL; both use L=96 and leave lr / batch at TSL defaults
(lr 0.0001, batch 32, 10 epochs, patience 3). That is not the official DLinear
script configuration (ECL official: batch 16, lr 0.001; Traffic official: lr 0.05).

The published L=96 point estimates already differ:

| Dataset | Horizon | iTransformer / PAC-Former Table 4 MSE | Official LTSF-Benchmark L=96 MSE |
|---|---|---|---|
| ECL | 96 | 0.197 | 0.141 |
| ECL | 720 | 0.245 | 0.210 |
| Traffic | 96 | 0.650 | 0.410 |
| Traffic | 720 | 0.645 | 0.464 |
| Weather | 96 | 0.196 | 0.182 |
| Exchange | 96 | 0.088 | 0.089 |

Traffic is the largest gap. Using official script learning rates at L=96 may land
closer to the official L=96 benchmark than to the numbers PAC-Former cited. If the
goal is to match Table 10 rather than official DLinear HPs, an alternative is the
TSL default recipe (lr 0.0001, batch 32) on all datasets. That choice is left open
until confirmed.

## Data

Same dataset bundle as the iTransformer reproduction (Google Drive id
`1l51QsKvQPcqILT3DwfjCgx8Dsg2rpjot`). Expected layout under `--data-root`:
`ETT-small/`, `electricity/`, `traffic/`, `weather/`, `exchange_rate/` (and `Solar/`,
unused here). Checksums: `data/SHA256SUMS`. Data files are not committed.

Official scripts use a flat `./dataset/*.csv` layout; the scheduler maps each job
to the corresponding subdirectory of this bundle.

## Environment

Python 3.11, torch 2.5.1+cu121, numpy 1.26.4, pandas 2.3.0, scikit-learn 1.7.0,
one RTX 4090 (24 GB). Official `requirements.txt` pins only `torch==1.9.0`.
GPU runs are not bit-deterministic (the official code does not enable cuDNN
determinism). DLinear is a pair of linear layers, so wall-clock time is dominated
by the test loop and data loading, not by training.

## How to run

```bash
python tools/build_configs.py                      # regenerate configs/jobs.csv (LOOKBACK=96)
python tools/scheduler.py --data-root /path/to/dataset --work-dir /path/to/work --dry-run
python tools/scheduler.py --data-root /path/to/dataset --work-dir /path/to/work        --max-parallel 2 --ram-budget-gb 60           # skip-completed, resumable
python tools/aggregate.py --results /path/to/work/results/results.csv --readme ../README.md
```

The scheduler gives each job its own working directory, reads test MSE/MAE from the
official log line `mse:..., mae:...` (official `metrics.npy` is commented out),
appends one row per seed to `results/results.csv`, and then deletes `pred.npy`,
`test_results/` PDFs and checkpoints. At most 2 jobs run together, and only while
the summed RAM estimate fits the budget (Traffic H720 holds all test predictions
in memory).

The 96 runs completed on 2026-11-20 (03:46–09:57 MYT) and did not overlap the iTransformer job.

## Results

The 96 runs finished on 2026-11-20, 03:46–09:57 MYT (6 h 11 m wall clock on one RTX 4090), with 0 failures. Numbers below are the 3-seed mean ± sample standard deviation (seeds 2026/2027/2028), never the best seed. "Paper" is the DLinear row of iTransformer Table 10 (arXiv:2310.06625v4), which PAC-Former Table 4 copies at lookback 96. Δ = (ours − paper) / paper; a negative value means a lower error than the paper.

Across the 32 dataset–horizon settings, mean MSE is 2.43% lower than the paper and mean MAE is 2.02% lower. ETTh2 H96 is pulled up by seed 2027 stopping early.

| Dataset | Horizon | Seeds | MSE (mean ± std) | MAE (mean ± std) | Paper MSE | Paper MAE | ΔMSE vs paper | ΔMAE vs paper |
|---|---|---|---|---|---|---|---|---|
| ETTh1 | 96 | 3 | 0.3859 ± 0.0039 | 0.3986 ± 0.0052 | 0.386 | 0.400 | -0.03% | -0.36% |
| ETTh1 | 192 | 3 | 0.4459 ± 0.0084 | 0.4403 ± 0.0094 | 0.437 | 0.432 | +2.04% | +1.91% |
| ETTh1 | 336 | 3 | 0.4955 ± 0.0161 | 0.4720 ± 0.0153 | 0.481 | 0.459 | +3.02% | +2.83% |
| ETTh1 | 720 | 3 | 0.5081 ± 0.0017 | 0.5029 ± 0.0020 | 0.519 | 0.516 | -2.11% | -2.54% |
| ETTh2 | 96 | 3 | 0.3679 ± 0.0885 | 0.4071 ± 0.0582 | 0.333 | 0.387 | +10.48% | +5.20% |
| ETTh2 | 192 | 3 | 0.4472 ± 0.0152 | 0.4515 ± 0.0113 | 0.477 | 0.476 | -6.26% | -5.14% |
| ETTh2 | 336 | 3 | 0.5159 ± 0.0363 | 0.4968 ± 0.0191 | 0.594 | 0.541 | -13.14% | -8.18% |
| ETTh2 | 720 | 3 | 0.6767 ± 0.0448 | 0.5866 ± 0.0181 | 0.831 | 0.657 | -18.57% | -10.71% |
| ETTm1 | 96 | 3 | 0.3456 ± 0.0002 | 0.3727 ± 0.0002 | 0.345 | 0.372 | +0.18% | +0.19% |
| ETTm1 | 192 | 3 | 0.3831 ± 0.0014 | 0.3932 ± 0.0020 | 0.380 | 0.389 | +0.82% | +1.08% |
| ETTm1 | 336 | 3 | 0.4151 ± 0.0014 | 0.4153 ± 0.0020 | 0.413 | 0.413 | +0.51% | +0.56% |
| ETTm1 | 720 | 3 | 0.4743 ± 0.0016 | 0.4526 ± 0.0024 | 0.474 | 0.453 | +0.07% | -0.09% |
| ETTm2 | 96 | 3 | 0.1862 ± 0.0005 | 0.2788 ± 0.0020 | 0.193 | 0.292 | -3.54% | -4.52% |
| ETTm2 | 192 | 3 | 0.2725 ± 0.0137 | 0.3464 ± 0.0122 | 0.284 | 0.362 | -4.04% | -4.31% |
| ETTm2 | 336 | 3 | 0.3492 ± 0.0200 | 0.3944 ± 0.0245 | 0.369 | 0.427 | -5.36% | -7.63% |
| ETTm2 | 720 | 3 | 0.5162 ± 0.0393 | 0.4931 ± 0.0326 | 0.554 | 0.522 | -6.83% | -5.53% |
| Exchange | 96 | 3 | 0.0777 ± 0.0012 | 0.1980 ± 0.0032 | 0.088 | 0.218 | -11.76% | -9.15% |
| Exchange | 192 | 3 | 0.1585 ± 0.0022 | 0.2935 ± 0.0027 | 0.176 | 0.315 | -9.92% | -6.84% |
| Exchange | 336 | 3 | 0.2994 ± 0.0062 | 0.4137 ± 0.0044 | 0.313 | 0.427 | -4.33% | -3.10% |
| Exchange | 720 | 3 | 0.7826 ± 0.0449 | 0.6715 ± 0.0180 | 0.839 | 0.695 | -6.72% | -3.38% |
| Weather | 96 | 3 | 0.1969 ± 0.0001 | 0.2557 ± 0.0006 | 0.196 | 0.255 | +0.44% | +0.29% |
| Weather | 192 | 3 | 0.2374 ± 0.0002 | 0.2942 ± 0.0002 | 0.237 | 0.296 | +0.17% | -0.61% |
| Weather | 336 | 3 | 0.2831 ± 0.0002 | 0.3318 ± 0.0003 | 0.283 | 0.335 | +0.03% | -0.95% |
| Weather | 720 | 3 | 0.3481 ± 0.0010 | 0.3844 ± 0.0013 | 0.345 | 0.381 | +0.91% | +0.89% |
| ECL | 96 | 3 | 0.1947 ± 0.0000 | 0.2776 ± 0.0001 | 0.197 | 0.282 | -1.17% | -1.56% |
| ECL | 192 | 3 | 0.1940 ± 0.0000 | 0.2803 ± 0.0002 | 0.196 | 0.285 | -1.04% | -1.63% |
| ECL | 336 | 3 | 0.2069 ± 0.0000 | 0.2961 ± 0.0003 | 0.209 | 0.301 | -0.98% | -1.63% |
| ECL | 720 | 3 | 0.2426 ± 0.0003 | 0.3295 ± 0.0007 | 0.245 | 0.333 | -0.96% | -1.05% |
| Traffic | 96 | 3 | 0.6492 ± 0.0011 | 0.3967 ± 0.0014 | 0.650 | 0.396 | -0.13% | +0.18% |
| Traffic | 192 | 3 | 0.5989 ± 0.0008 | 0.3710 ± 0.0009 | 0.598 | 0.370 | +0.15% | +0.27% |
| Traffic | 336 | 3 | 0.6059 ± 0.0007 | 0.3742 ± 0.0008 | 0.605 | 0.373 | +0.15% | +0.33% |
| Traffic | 720 | 3 | 0.6463 ± 0.0006 | 0.3956 ± 0.0006 | 0.645 | 0.394 | +0.20% | +0.42% |

Average relative difference over the four horizons in each dataset:

| Dataset | ΔMSE vs paper | ΔMAE vs paper |
|---|---|---|
| ETTh1 | +0.73% | +0.46% |
| ETTh2 | -6.87% | -4.71% |
| ETTm1 | +0.39% | +0.44% |
| ETTm2 | -4.94% | -5.50% |
| Exchange | -8.18% | -5.62% |
| Weather | +0.39% | -0.09% |
| ECL | -1.04% | -1.47% |
| Traffic | +0.10% | +0.30% |
| **Overall** | **-2.43%** | **-2.02%** |

## Files

* `official/` official code (Apache-2.0, see `THIRD_PARTY_NOTICE.md`)
* `patches/official_changes.diff` every change made to the official code
* `configs/jobs.csv` job list; `configs/official_script_args.json` parsed scripts + defaults
* `configs/paper_reference.csv` PAC-Former / iTransformer Table 10 numbers, plus official L=96 / best-L columns
* `tools/` config builder, scheduler, aggregation, launch script
* `results/` per-seed `results.csv` and `summary.csv` (filled after the runs)
