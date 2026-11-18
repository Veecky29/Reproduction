# iTransformer reproduction

Faithful re-run of **iTransformer** (Liu et al., ICLR 2024, arXiv:2310.06625) with the
authors' official code and official scripts, 3 seeds (2026, 2027, 2028).

## What is reproduced

* Long-term forecasting benchmark of the paper (Table 10, averages in Table 1):
  ETTh1, ETTh2, ETTm1, ETTm2, Exchange, Weather, ECL (Electricity), Solar-Energy, Traffic;
  lookback 96, horizons 96 / 192 / 336 / 720. 9 datasets x 4 horizons x 3 seeds = **108 runs**
  (`configs/jobs.csv`).
* PEMS03/04/07/08 (paper Table 9, horizons 12 / 24 / 48 / 96) are prepared in
  `configs/jobs_pems.csv` (48 runs) but are not part of the default run.

## Protocol (= official code, unchanged)

* Code: `official/` = thuml/iTransformer @ `c2426e68ca13f74aaec08045c5c724d8ad328124`.
  Only change: `--seed` argument in `run.py` (see `patches/official_changes.diff`).
* Hyperparameters: taken verbatim from `official/scripts/multivariate_forecasting/*/*.sh`;
  anything a script does not set falls back to the `run.py` default. `tools/build_configs.py`
  generates `configs/jobs.csv` from those files automatically (no hand-typed values; the
  `*_src` columns say where each value came from). No hyperparameter search.
* Fixed by the official code: Adam, MSE loss, `train_epochs` 10, early stopping on validation MSE with
  patience 3, `lradj=type1` (lr halves every epoch), dropout 0.1, n_heads 8, RevIN-style
  normalisation (`use_norm`), timeF covariates as extra tokens, train/val loaders shuffle=True and
  drop_last=True, **test loader batch_size=1** (so every test window is evaluated), metrics computed
  on the normalised scale.
* Seeds: 2026, 2027, 2028 (`--seed`); the paper's Table 5 reports 5 seeds, the official script uses 2023.
* Main numbers are **mean ± sample std over the 3 seeds**, never the best seed.

## Hyperparameters (from the official scripts; seed-independent)

| Dataset | Horizon | e_layers | d_model | d_ff | n_heads | batch | lr | epochs | patience | lradj | use_norm | Official script |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ETTh1 | 96 | 2 | 256 | 256 | 8 | 32 | 0.0001 | 10 | 3 | type1 | 1 | `ETT/iTransformer_ETTh1.sh` |
| ETTh1 | 192 | 2 | 256 | 256 | 8 | 32 | 0.0001 | 10 | 3 | type1 | 1 | `ETT/iTransformer_ETTh1.sh` |
| ETTh1 | 336 | 2 | 512 | 512 | 8 | 32 | 0.0001 | 10 | 3 | type1 | 1 | `ETT/iTransformer_ETTh1.sh` |
| ETTh1 | 720 | 2 | 512 | 512 | 8 | 32 | 0.0001 | 10 | 3 | type1 | 1 | `ETT/iTransformer_ETTh1.sh` |
| ETTh2 | 96 | 2 | 128 | 128 | 8 | 32 | 0.0001 | 10 | 3 | type1 | 1 | `ETT/iTransformer_ETTh2.sh` |
| ETTh2 | 192 | 2 | 128 | 128 | 8 | 32 | 0.0001 | 10 | 3 | type1 | 1 | `ETT/iTransformer_ETTh2.sh` |
| ETTh2 | 336 | 2 | 128 | 128 | 8 | 32 | 0.0001 | 10 | 3 | type1 | 1 | `ETT/iTransformer_ETTh2.sh` |
| ETTh2 | 720 | 2 | 128 | 128 | 8 | 32 | 0.0001 | 10 | 3 | type1 | 1 | `ETT/iTransformer_ETTh2.sh` |
| Exchange | 96 | 2 | 128 | 128 | 8 | 32 | 0.0001 | 10 | 3 | type1 | 1 | `Exchange/iTransformer.sh` |
| Exchange | 192 | 2 | 128 | 128 | 8 | 32 | 0.0001 | 10 | 3 | type1 | 1 | `Exchange/iTransformer.sh` |
| Exchange | 336 | 2 | 128 | 128 | 8 | 32 | 0.0001 | 1 | 3 | type1 | 1 | `Exchange/iTransformer.sh` |
| Exchange | 720 | 2 | 128 | 128 | 8 | 32 | 0.0001 | 10 | 3 | type1 | 1 | `Exchange/iTransformer.sh` |
| ETTm1 | 96 | 2 | 128 | 128 | 8 | 32 | 0.0001 | 10 | 3 | type1 | 1 | `ETT/iTransformer_ETTm1.sh` |
| ETTm1 | 192 | 2 | 128 | 128 | 8 | 32 | 0.0001 | 10 | 3 | type1 | 1 | `ETT/iTransformer_ETTm1.sh` |
| ETTm1 | 336 | 2 | 128 | 128 | 8 | 32 | 0.0001 | 10 | 3 | type1 | 1 | `ETT/iTransformer_ETTm1.sh` |
| ETTm1 | 720 | 2 | 128 | 128 | 8 | 32 | 0.0001 | 10 | 3 | type1 | 1 | `ETT/iTransformer_ETTm1.sh` |
| ETTm2 | 96 | 2 | 128 | 128 | 8 | 32 | 0.0001 | 10 | 3 | type1 | 1 | `ETT/iTransformer_ETTm2.sh` |
| ETTm2 | 192 | 2 | 128 | 128 | 8 | 32 | 0.0001 | 10 | 3 | type1 | 1 | `ETT/iTransformer_ETTm2.sh` |
| ETTm2 | 336 | 2 | 128 | 128 | 8 | 32 | 0.0001 | 10 | 3 | type1 | 1 | `ETT/iTransformer_ETTm2.sh` |
| ETTm2 | 720 | 2 | 128 | 128 | 8 | 32 | 0.0001 | 10 | 3 | type1 | 1 | `ETT/iTransformer_ETTm2.sh` |
| Weather | 96 | 3 | 512 | 512 | 8 | 32 | 0.0001 | 10 | 3 | type1 | 1 | `Weather/iTransformer.sh` |
| Weather | 192 | 3 | 512 | 512 | 8 | 32 | 0.0001 | 10 | 3 | type1 | 1 | `Weather/iTransformer.sh` |
| Weather | 336 | 3 | 512 | 512 | 8 | 32 | 0.0001 | 10 | 3 | type1 | 1 | `Weather/iTransformer.sh` |
| Weather | 720 | 3 | 512 | 512 | 8 | 32 | 0.0001 | 10 | 3 | type1 | 1 | `Weather/iTransformer.sh` |
| ECL | 96 | 3 | 512 | 512 | 8 | 16 | 0.0005 | 10 | 3 | type1 | 1 | `ECL/iTransformer.sh` |
| ECL | 192 | 3 | 512 | 512 | 8 | 16 | 0.0005 | 10 | 3 | type1 | 1 | `ECL/iTransformer.sh` |
| ECL | 336 | 3 | 512 | 512 | 8 | 16 | 0.0005 | 10 | 3 | type1 | 1 | `ECL/iTransformer.sh` |
| ECL | 720 | 3 | 512 | 512 | 8 | 16 | 0.0005 | 10 | 3 | type1 | 1 | `ECL/iTransformer.sh` |
| Solar-Energy | 96 | 2 | 512 | 512 | 8 | 32 | 0.0005 | 10 | 3 | type1 | 1 | `SolarEnergy/iTransformer.sh` |
| Solar-Energy | 192 | 2 | 512 | 512 | 8 | 32 | 0.0005 | 10 | 3 | type1 | 1 | `SolarEnergy/iTransformer.sh` |
| Solar-Energy | 336 | 2 | 512 | 512 | 8 | 32 | 0.0005 | 10 | 3 | type1 | 1 | `SolarEnergy/iTransformer.sh` |
| Solar-Energy | 720 | 2 | 512 | 512 | 8 | 32 | 0.0005 | 10 | 3 | type1 | 1 | `SolarEnergy/iTransformer.sh` |
| Traffic | 96 | 4 | 512 | 512 | 8 | 16 | 0.001 | 10 | 3 | type1 | 1 | `Traffic/iTransformer.sh` |
| Traffic | 192 | 4 | 512 | 512 | 8 | 16 | 0.001 | 10 | 3 | type1 | 1 | `Traffic/iTransformer.sh` |
| Traffic | 336 | 4 | 512 | 512 | 8 | 16 | 0.001 | 10 | 3 | type1 | 1 | `Traffic/iTransformer.sh` |
| Traffic | 720 | 4 | 512 | 512 | 8 | 16 | 0.001 | 10 | 3 | type1 | 1 | `Traffic/iTransformer.sh` |
| PEMS08 | 12 | 2 | 512 | 512 | 8 | 32 | 0.0001 | 10 | 3 | type1 | 1 | `PEMS/iTransformer_08.sh` |
| PEMS08 | 24 | 2 | 512 | 512 | 8 | 32 | 0.0001 | 10 | 3 | type1 | 1 | `PEMS/iTransformer_08.sh` |
| PEMS08 | 48 | 4 | 512 | 512 | 8 | 16 | 0.001 | 10 | 3 | type1 | 0 | `PEMS/iTransformer_08.sh` |
| PEMS08 | 96 | 4 | 512 | 512 | 8 | 16 | 0.001 | 10 | 3 | type1 | 0 | `PEMS/iTransformer_08.sh` |
| PEMS04 | 12 | 4 | 1024 | 1024 | 8 | 32 | 0.0005 | 10 | 3 | type1 | 0 | `PEMS/iTransformer_04.sh` |
| PEMS04 | 24 | 4 | 1024 | 1024 | 8 | 32 | 0.0005 | 10 | 3 | type1 | 0 | `PEMS/iTransformer_04.sh` |
| PEMS04 | 48 | 4 | 1024 | 1024 | 8 | 32 | 0.0005 | 10 | 3 | type1 | 0 | `PEMS/iTransformer_04.sh` |
| PEMS04 | 96 | 4 | 1024 | 1024 | 8 | 32 | 0.0005 | 10 | 3 | type1 | 0 | `PEMS/iTransformer_04.sh` |
| PEMS03 | 12 | 4 | 512 | 512 | 8 | 32 | 0.001 | 10 | 3 | type1 | 1 | `PEMS/iTransformer_03.sh` |
| PEMS03 | 24 | 4 | 512 | 512 | 8 | 32 | 0.001 | 10 | 3 | type1 | 1 | `PEMS/iTransformer_03.sh` |
| PEMS03 | 48 | 4 | 512 | 512 | 8 | 32 | 0.001 | 10 | 3 | type1 | 1 | `PEMS/iTransformer_03.sh` |
| PEMS03 | 96 | 4 | 512 | 512 | 8 | 32 | 0.001 | 10 | 3 | type1 | 1 | `PEMS/iTransformer_03.sh` |
| PEMS07 | 12 | 2 | 512 | 512 | 8 | 32 | 0.001 | 10 | 3 | type1 | 0 | `PEMS/iTransformer_07.sh` |
| PEMS07 | 24 | 2 | 512 | 512 | 8 | 32 | 0.001 | 10 | 3 | type1 | 0 | `PEMS/iTransformer_07.sh` |
| PEMS07 | 48 | 4 | 512 | 512 | 8 | 16 | 0.001 | 10 | 3 | type1 | 0 | `PEMS/iTransformer_07.sh` |
| PEMS07 | 96 | 4 | 512 | 512 | 8 | 16 | 0.001 | 10 | 3 | type1 | 0 | `PEMS/iTransformer_07.sh` |

## Paper text vs official scripts

The appendix says "batch size uniformly 32, epochs fixed to 10, L ∈ {2,3,4}, D ∈ {256, 512},
lr ∈ {1e-3, 5e-4, 1e-4}". The official scripts deviate in places; we follow the scripts, because
they are the authors' released configuration for the reported numbers:

* batch size 16 for ECL, Traffic, PEMS07/PEMS08 (H48, H96);
* d_model = d_ff = 128 for ETTh2, ETTm1, ETTm2, Exchange and 1024 for PEMS04 (outside {256, 512});
* **Exchange H336: `--train_epochs 1`** (only script line that changes the epoch count);
* `use_norm 0` for PEMS04, PEMS07 and PEMS08 (H48, H96).

## Data

Official dataset bundle linked from the iTransformer README (Google Drive id
`1l51QsKvQPcqILT3DwfjCgx8Dsg2rpjot`, zip sha256
`01d7016eaa3eb571b77542612219d116dfc6a9f5399586dea1f98261fc4d80c4`; the Tsinghua Cloud link in the
README is dead as of 2026-11-19). Expected layout under `--data-root`: `ETT-small/`, `electricity/`,
`traffic/`, `weather/`, `exchange_rate/`, `Solar/`, `PEMS/`. Checksums of every file:
`data/SHA256SUMS`. Data files are not committed.

## Environment

Python 3.11, torch 2.5.1+cu121, numpy 1.26.4, pandas 2.3.0, scikit-learn 1.7.0,
matplotlib 3.9.4, one RTX 4090 (24 GB). The official `requirements.txt` pins torch 2.0.0 / pandas 1.5.3;
`reformer-pytorch==1.4.4` (imported but not used by iTransformer) is installed with
einops 0.6.1, axial-positional-embedding 0.2.1, local-attention 1.8.6, product-key-memory 0.1.10.
GPU runs are not bit-deterministic (the official code does not enable cuDNN determinism).

## How to run

```bash
python tools/build_configs.py                      # regenerate configs/jobs*.csv from the official scripts
python tools/scheduler.py --data-root /path/to/dataset --work-dir /path/to/work --dry-run
python tools/scheduler.py --data-root /path/to/dataset --work-dir /path/to/work \
       --max-parallel 2 --ram-budget-gb 60           # skip-completed, resumable
python tools/aggregate.py --results /path/to/work/results/results.csv --readme ../README.md
```

The scheduler gives each job its own working directory (the official code writes
`./checkpoints`, `./results` and `./test_results` relative to the CWD), reads the test MSE/MAE from
the official `metrics.npy`, appends one row per seed to `results/results.csv`, and then deletes the
large `pred.npy` / `true.npy` / checkpoint files. At most 2 jobs run together, and only while the
summed RAM estimate fits the budget (the official test loop holds all predictions in memory; Traffic
H720 needs ~30 GB, so those runs go one at a time).

## Files

* `official/` official code (MIT, see `THIRD_PARTY_NOTICE.md`)
* `patches/official_changes.diff` every change made to the official code
* `configs/jobs.csv`, `configs/jobs_pems.csv` job lists; `configs/official_script_args.json` parsed scripts + run.py defaults
* `configs/paper_reference.csv` paper MSE/MAE (Tables 9, 10) and 5-seed std (Table 5)
* `tools/` config builder, scheduler, aggregation, launch script
* `results/` per-seed `results.csv` and `summary.csv` (filled after the runs)
