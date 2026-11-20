# RLinear reproduction

Faithful re-run of **RLinear** (Li et al., arXiv:2305.10721; official
code: [plumprc/RTSF](https://github.com/plumprc/RTSF)) with the authors'
official code and official script hyperparameters, 3 seeds (2026, 2027, 2028).

**Status: code ready, not yet run.** Do not start this while the PatchTST
reproduction is still using the GPU.

This is the next baseline in the PAC-Former comparison after PatchTST.
PAC-Former Table 4 copies RLinear's numbers from iTransformer (Liu et al.,
ICLR 2024) Appendix F.4 Table 10 at **lookback L = 96**, not from the official
RTSF scripts, which hard-code lookback 336 and only list `pred_len` 96.

## What is reproduced

* Long-term multivariate forecasting, lookback **96**, horizons 96 / 192 / 336 / 720.
  Datasets: ETTh1, ETTh2, ETTm1, ETTm2, Exchange, Weather, ECL (Electricity), Traffic.
  8 datasets × 4 horizons × 3 seeds = **96 runs** (`configs/jobs.csv`).
* PAC-Former Table 4 uses 7 of these (no Exchange). Exchange is included because
  it is in iTransformer Table 10.
* **Exchange has no official RTSF script.** Jobs use the ETTh1 official recipe
  (batch 128, lr 0.005, dropout 0.1, RevIN on, shared linear). This is a
  **non-official choice**, flagged `proposed=1` in `configs/jobs.csv` and
  recorded in `configs/official_script_args.json`.
* **Traffic has no official RTSF script.** Jobs use the ECL official recipe
  (batch 16, lr 0.001, dropout 0, RevIN on, shared linear). This is a
  **non-official choice**, flagged `proposed=1`.
* **Solar-Energy is not in the default job list.** There is no official RTSF
  script for it. Table 10 numbers are recorded in `configs/paper_reference.csv`
  for reference only.
* ILI is not part of this run.

## Protocol

* Code: `official/` = plumprc/RTSF @ `0fec00104f754f4fbf795b9b4da5fa2459b32e76`
  (2023-09-13). Two minimal changes, see `patches/official_changes.diff`:
  * `--seed` argument in `run.py` (official default 1024, hard-coded before argparse).
  * `--model Linear` is aliased to `RLinear` in `exp/exp_main.py`. Official
    `script.md` passes `--model Linear`, but the model dictionary at this commit
    only registers `RLinear`. Jobs emit `--model RLinear`.
  Upstream declares no license; see `THIRD_PARTY_NOTICE.md`.
* Hyperparameters: taken from `official/script.md` except **`seq_len` is set to
  96** (the official commands hard-code 336) and each official `pred_len=96`
  example is expanded to 96 / 192 / 336 / 720 with the same hyperparameters.
  Anything a script does not set falls back to the `run.py` default.
  `tools/build_configs.py` generates `configs/jobs.csv` from `script.md` (the
  `*_src` columns say where each value came from). No hyperparameter search.
  `LOOKBACK=336 python tools/build_configs.py` will emit a job list with the
  official lookback (horizons still expanded) if that protocol is chosen later.
* Model: a single shared linear layer (not `--individual`) plus RevIN
  (`--rev`). Dropout is 0.1 unless the script sets `--drop 0` (ECL, and the
  proposed Traffic recipe). Loss is MSE. Optimizer is Adam, `lradj=type1`
  (lr halves every epoch), `train_epochs` 15, early stopping on validation MSE
  with patience 10. These last three are `run.py` defaults; `script.md` does
  not set them.
* Official test loader uses the training `batch_size` with `drop_last=True`, so
  the last incomplete test batch is dropped. Official `test()` prints
  `mse:, mae:, R2:` to four decimal places and keeps preds / trues / inputx in
  RAM. Metrics are computed on the standardised scale.
* Seeds: 2026, 2027, 2028 (`--seed`); official `script.md` uses `--itr 1` and
  hard-codes 1024.
* Main numbers are **mean ± sample std over the 3 seeds**, never the best seed.
* Primary reference values (`paper_mse` / `paper_mae`) are PAC-Former Table 4 =
  iTransformer Table 10 (L=96), RLinear column. Those numbers were produced by
  the iTransformer authors at L=96 and need not match a run of the official
  RTSF scripts (which use L=336).

## Lookback 96 vs 336

The official RTSF `script.md` commands and the RLinear paper's reported
lookback use 336. PAC-Former and iTransformer Table 10 fix L=96 for every
baseline, and PAC-Former §4.5 says it uses those fixed-L96 values "rather than
substituting results obtained with a different or optimized lookback".

This reproduction follows L=96 and official script learning rates, batch sizes
and dropout, because the goal is the baseline PAC-Former actually compared
against.

## Hyperparameters (official scripts, seq_len overridden to 96; seed-independent)

| Dataset | Horizon | channel | batch | lr | dropout | epochs | patience | lradj | RevIN | individual | Official script | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ETTh1 | 96 / 192 / 336 / 720 | 7 | 128 | 0.005 | 0.1 | 15 | 10 | type1 | yes | no | `script.md` | drop / epochs / patience / lradj from `run.py` defaults |
| ETTh2 | 96 / 192 / 336 / 720 | 7 | **256** | 0.005 | 0.1 | 15 | 10 | type1 | yes | no | `script.md` | same defaults |
| ETTm1 | 96 / 192 / 336 / 720 | 7 | 128 | 0.005 | 0.1 | 15 | 10 | type1 | yes | no | `script.md` | same defaults |
| ETTm2 | 96 / 192 / 336 / 720 | 7 | 128 | 0.005 | 0.1 | 15 | 10 | type1 | yes | no | `script.md` | same defaults |
| Exchange | 96 / 192 / 336 / 720 | 8 | 128 | 0.005 | 0.1 | 15 | 10 | type1 | yes | no | **none** | **non-official:** follows ETTh1 |
| Weather | 96 / 192 / 336 / 720 | 21 | 128 | 0.005 | 0.1 | 15 | 10 | type1 | yes | no | `script.md` | `Weather.csv` mapped to `weather.csv` |
| ECL | 96 / 192 / 336 / 720 | 321 | **16** | **0.001** | **0** | 15 | 10 | type1 | yes | no | `script.md` | batch is the `run.py` default (script omits it); `ECL.csv` mapped to `electricity.csv` |
| Traffic | 96 / 192 / 336 / 720 | 862 | 16 | 0.001 | 0 | 15 | 10 | type1 | yes | no | **none** | **non-official:** follows ECL |

## Data

Same dataset bundle as the iTransformer reproduction (Google Drive id
`1l51QsKvQPcqILT3DwfjCgx8Dsg2rpjot`). Expected layout under `--data-root`:
`ETT-small/`, `electricity/`, `traffic/`, `weather/`, `exchange_rate/` (and `Solar/`,
unused here). Checksums: `data/SHA256SUMS`. Data files are not committed.

Official `script.md` uses a flat layout with `Weather.csv` and `ECL.csv`. The
scheduler maps each job to the corresponding subdirectory of this bundle and
to `weather.csv` / `electricity.csv`.

## Environment

Python 3.11, torch 2.5.1+cu121, numpy / pandas / scikit-learn / matplotlib,
one RTX 4090 (24 GB). Official RTSF does not ship a `requirements.txt`.
GPU runs are not bit-deterministic (the official code does not enable cuDNN
determinism). RLinear is a single linear layer, so wall-clock time is dominated
by data loading and the official test loop (which holds all test predictions
in memory). Traffic H720 is estimated at about 46 GB RAM; two such jobs will
not be co-scheduled under the default 60 GB budget.

## How to run

```bash
python tools/build_configs.py                      # regenerate configs/jobs.csv (LOOKBACK=96)
python tools/scheduler.py --data-root /path/to/dataset --work-dir /path/to/work --dry-run
python tools/scheduler.py --data-root /path/to/dataset --work-dir /path/to/work \
          --max-parallel 2 --ram-budget-gb 60 --num-workers 10   # skip-completed, resumable
python tools/aggregate.py --results /path/to/work/results/results.csv --readme ../README.md
```

`--max-parallel` and `--num-workers` are configurable. Official `num_workers`
is 10. When sharing the GPU with another training job, use `--num-workers 2`
(and consider `--max-parallel 1`) so dataloader workers do not slow the other
run. Suggested order if launching by hand: ETTh1, ETTh2, Exchange, ETTm1,
ETTm2, Weather, then ECL, with Traffic last. `--datasets ETTh1,ETTh2,Exchange,ETTm1,ETTm2,Weather`
restricts the queue.

Or, on the 4090 VM (only in a **new** directory, after PatchTST has finished):

```bash
# default: official num_workers=10, max-parallel 2
bash tools/launch.sh
# if the GPU must be shared (not recommended while PatchTST ECL is running):
NUM_WORKERS=2 MAX_PARALLEL=1 bash tools/launch.sh
```

`tools/launch.sh` defaults to `$HOME/PACFormer/repro_rlinear_20261121/` and the
shared iTransformer dataset bundle. Do not modify existing VM trees.

The scheduler gives each job its own working directory, reads test MSE/MAE from
the official log line `mse:..., mae:..., R2:...`, appends one row per seed to
`results/results.csv`, writes `results/status.json`, and then deletes
`test_results/` PDFs and checkpoints. At most 2 jobs run together, and only
while the summed RAM estimate fits the budget.

## Files

* `official/` official RTSF code (no upstream license declared; see
  `THIRD_PARTY_NOTICE.md`). `pics/` and `tutorial.ipynb` were not copied.
* `patches/official_changes.diff` the `--seed` and `Linear` → `RLinear` changes
* `configs/jobs.csv` 96-job list; `configs/official_script_args.json` parsed
  `script.md`, defaults, and the Exchange / Traffic proposals
* `configs/paper_reference.csv` iTransformer Table 10 RLinear column (L=96)
* `tools/` config builder, scheduler, aggregation, launch script
* `results/` per-seed `results.csv` and `summary.csv` (filled after the runs)
