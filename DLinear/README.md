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

Do not start this while the iTransformer reproduction is still using the GPU.

## Files

* `official/` official code (Apache-2.0, see `THIRD_PARTY_NOTICE.md`)
* `patches/official_changes.diff` every change made to the official code
* `configs/jobs.csv` job list; `configs/official_script_args.json` parsed scripts + defaults
* `configs/paper_reference.csv` PAC-Former / iTransformer Table 10 numbers, plus official L=96 / best-L columns
* `tools/` config builder, scheduler, aggregation, launch script
* `results/` per-seed `results.csv` and `summary.csv` (filled after the runs)
