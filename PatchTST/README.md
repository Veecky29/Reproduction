# PatchTST reproduction

Faithful re-run of **PatchTST** (Nie et al., ICLR 2023, arXiv:2211.14730; official
code: [yuqinie98/PatchTST](https://github.com/yuqinie98/PatchTST)) with the
authors' official supervised code and official script hyperparameters, 3 seeds
(2026, 2027, 2028).

This is the next baseline in the PAC-Former comparison after DLinear. PAC-Former
Table 4 copies PatchTST's numbers from iTransformer (Liu et al., ICLR 2024)
Appendix F.4 Table 10 at **lookback L = 96**, not from the PatchTST paper's
default lookback of 336 (PatchTST/42).

## What is reproduced

* Long-term multivariate forecasting, lookback **96**, horizons 96 / 192 / 336 / 720.
  Datasets scheduled: ETTh1, ETTh2, ETTm1, ETTm2, Weather, ECL (Electricity).
  6 datasets × 4 horizons × 3 seeds = **72 runs** (`configs/jobs.csv`).
* PAC-Former Table 4 covers these 6 datasets plus Traffic (no Exchange).
* **Exchange is not in the default job list.** There is no official PatchTST
  script. An ETTh-style proposal is recorded in `configs/official_script_args.json`
  and in the table below, but it is not scheduled. `INCLUDE_EXCHANGE=1 python tools/build_configs.py`
  will add the 12 Exchange jobs if that protocol is chosen later.
* **Traffic is not in the default job list.** An official script exists
  (`scripts/PatchTST/traffic.sh`) and its hyperparameters are recorded below and
  in `configs/official_script_args.json`, but Traffic is left unscheduled because
  the VM rental ends around 2026-11-22 and Traffic (862 channels, channel-independent)
  would dominate wall-clock time. `INCLUDE_TRAFFIC=1 python tools/build_configs.py`
  will add the 12 Traffic jobs if that protocol is chosen later.
* **Solar-Energy is not in the default job list.** There is no official PatchTST
  script for it. Table 10 numbers are recorded in `configs/paper_reference.csv`
  for reference only.
* ILI / illness is not part of this run.

## Protocol

* Code: `official/` = yuqinie98/PatchTST `PatchTST_supervised/` @
  `204c21efe0b39603ad6e2ca640ef5896646ab1a9` (Apache-2.0).
  **No code changes.** Official `run_longExp.py` already exposes `--random_seed`
  (default 2021). See `patches/official_changes.diff` (documentation only).
* Hyperparameters: taken verbatim from `official/scripts/PatchTST/{etth1,etth2,ettm1,ettm2,weather,electricity,traffic}.sh`
  except **`seq_len` is set to 96** (the official scripts hard-code 336). Anything a
  script does not set falls back to the `run_longExp.py` default. `tools/build_configs.py`
  generates `configs/jobs.csv` from those files (the `*_src` columns say where each
  value came from). No hyperparameter search. `LOOKBACK=336 python tools/build_configs.py`
  will emit a verbatim official-lookback job list if that protocol is chosen later.
* Model: PatchTST (channel-independent Transformer on patches), RevIN on
  (`--revin 1`), no decomposition, shared (not individual) head. Official scripts
  all use `--patch_len 16 --stride 8` and `--e_layers 3`. With end-padding this
  is 12 patches at L=96, versus 42 patches at the official L=336 (PatchTST/42).
* Fixed by the official code: Adam, MSE loss, `train_epochs` 100. Early stopping
  is on validation MSE; **patience is 100 (the argparse default) for ETTh1 / ETTh2**,
  so those jobs almost always run all 100 epochs. ETTm / Weather use patience 20;
  ECL (and the unscheduled Traffic script) use patience 10. Default `lradj` is
  `type3` (lr held for 2 epochs, then ×0.9 per epoch); ETTm and ECL use
  `lradj=TST` (OneCycleLR, with script `pct_start`).
* Test loader (official `data_factory.py`): uses the **training `batch_size`**
  and **`drop_last=True`**, so the trailing incomplete test batch is discarded.
  Metrics are computed on the standardised scale. PatchTST does not consume time
  features (`forward` takes `batch_x` only); `--freq` / `--embed` are left at
  official defaults.
* Seeds: 2026, 2027, 2028 (`--random_seed`); official scripts use `--itr 1` and
  `--random_seed 2021`.
* Main numbers are **mean ± sample std over the 3 seeds**, never the best seed.
* Primary reference values (`paper_mse` / `paper_mae`) are PAC-Former Table 4 =
  iTransformer Table 10 (L=96), PatchTST column. The official PatchTST paper
  reports L=336 (and omits Exchange); those numbers are not used as the L=96 target.

## Lookback 96 vs 336

The official supervised scripts and the PatchTST paper's main results use lookback
336. PAC-Former and iTransformer Table 10 fix L=96 for every baseline, and
PAC-Former §4.5 says it uses those fixed-L96 values "rather than substituting
results obtained with a different or optimized lookback".

This reproduction follows L=96, official script learning rates, batch sizes,
d_model, dropout and schedules, because the goal is the baseline PAC-Former
actually compared against.

## Hyperparameters (official scripts, seq_len overridden to 96; seed-independent)

| Dataset | Horizon | enc_in | patch / stride | d_model / n_heads / d_ff | dropout / fc / head | batch | lr | epochs / patience | lradj (pct_start) | Official script | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ETTh1 | 96 / 192 / 336 / 720 | 7 | 16 / 8 | 16 / 4 / 128 | 0.3 / 0.3 / 0 | 128 | 0.0001 | 100 / **100** (default) | type3 (default; 0.3 unused) | `PatchTST/etth1.sh` | Will run ~100 epochs |
| ETTh2 | 96 / 192 / 336 / 720 | 7 | 16 / 8 | 16 / 4 / 128 | 0.3 / 0.3 / 0 | 128 | 0.0001 | 100 / **100** (default) | type3 (default; 0.3 unused) | `PatchTST/etth2.sh` | Will run ~100 epochs |
| Exchange | 96 / 192 / 336 / 720 | 8 | 16 / 8 | 16 / 4 / 128 | 0.3 / 0.3 / 0 | 128 | 0.0001 | 100 / **100** (default) | type3 (default) | *none — proposed from etth1.sh* | **Not scheduled** |
| ETTm1 | 96 / 192 / 336 / 720 | 7 | 16 / 8 | 128 / 16 / 256 | 0.2 / 0.2 / 0 | 128 | 0.0001 | 100 / 20 | TST (0.4) | `PatchTST/ettm1.sh` | |
| ETTm2 | 96 / 192 / 336 / 720 | 7 | 16 / 8 | 128 / 16 / 256 | 0.2 / 0.2 / 0 | 128 | 0.0001 | 100 / 20 | TST (0.4) | `PatchTST/ettm2.sh` | |
| Weather | 96 / 192 / 336 / 720 | 21 | 16 / 8 | 128 / 16 / 256 | 0.2 / 0.2 / 0 | 128 | 0.0001 | 100 / 20 | type3 (default; 0.3 unused) | `PatchTST/weather.sh` | lradj not set in script |
| ECL | 96 / 192 / 336 / 720 | 321 | 16 / 8 | 128 / 16 / 256 | 0.2 / 0.2 / 0 | 32 | 0.0001 | 100 / 10 | TST (0.2) | `PatchTST/electricity.sh` | Run last; see concurrency |
| Traffic | 96 / 192 / 336 / 720 | 862 | 16 / 8 | 128 / 16 / 256 | 0.2 / 0.2 / 0 | 24 | 0.0001 | 100 / 10 | TST (0.2) | `PatchTST/traffic.sh` | **Not scheduled** |

Shared official defaults (not set by these scripts unless noted): `--revin 1`,
`--affine 0`, `--decomposition 0`, `--individual 0`, `--padding_patch end`,
`--itr 1` (scripts set this). Parameter counts at L=96 (shared head, channel-independent):
ETTh / Exchange ≈ 0.035M (H96) / 0.156M (H720); ETTm / Weather / ECL / Traffic
≈ 0.549M (H96) / 1.508M (H720). Counts do not grow with the number of channels.

## Exchange (not scheduled)

There is no `scripts/PatchTST/exchange_rate.sh`. The official PatchTST paper
explicitly omits the Exchange-rate dataset (Appendix; financial series close to
a random-walk baseline). Exchange is in iTransformer Table 10 but is **not**
in the default `jobs.csv`.

A documented proposal (ETTh-style small model: `enc_in=8`, `d_model=16`,
`n_heads=4`, `d_ff=128`, `dropout=0.3`, batch 128, lr 0.0001, 100 epochs,
patience 100 / `lradj=type3` defaults) is stored in
`configs/official_script_args.json` under `proposed_exchange`. The main
alternative considered was ETTm-style (`d_model=128`, `lradj=TST`,
`patience=20`). `INCLUDE_EXCHANGE=1 python tools/build_configs.py` will emit
those 12 jobs if they are scheduled later.

## Official test `drop_last=True`

Unlike the iTransformer official loader (test `batch_size=1`, `drop_last=False`),
PatchTST evaluates the test set in training-sized batches and drops the remainder.
Dropped windows at L=96 (kept as official behaviour):

| Dataset | H96 | H192 | H336 | H720 |
|---|---|---|---|---|
| ETTh1 / ETTh2 | 97 / 2785 | 1 / 2689 | 113 / 2545 | 113 / 2161 |
| ETTm1 / ETTm2 | 33 / 11425 | 65 / 11329 | 49 / 11185 | 49 / 10801 |
| Exchange (not scheduled) | 14 / 1422 | 46 / 1326 | 30 / 1182 | 30 / 798 |
| Weather | 76 / 10444 | 108 / 10348 | 92 / 10204 | 92 / 9820 |
| ECL | 13 / 5165 | 13 / 5069 | 29 / 4925 | 29 / 4541 |

## Paper reference (iTransformer Table 10, PatchTST column, L=96)

Values verified from arXiv:2310.06625 Table 10 (full long-term forecasting
results; input length 96). PAC-Former Table 4 copies these for the datasets it
includes. Empty cells are not used; nothing was invented.

| Dataset | H96 MSE / MAE | H192 | H336 | H720 |
|---|---|---|---|---|
| ETTh1 | 0.414 / 0.419 | 0.460 / 0.445 | 0.501 / 0.466 | 0.500 / 0.488 |
| ETTh2 | 0.302 / 0.348 | 0.388 / 0.400 | 0.426 / 0.433 | 0.431 / 0.446 |
| ETTm1 | 0.329 / 0.367 | 0.367 / 0.385 | 0.399 / 0.410 | 0.454 / 0.439 |
| ETTm2 | 0.175 / 0.259 | 0.241 / 0.302 | 0.305 / 0.343 | 0.402 / 0.400 |
| Exchange (not scheduled) | 0.088 / 0.205 | 0.176 / 0.299 | 0.301 / 0.397 | 0.901 / 0.714 |
| Weather | 0.177 / 0.218 | 0.225 / 0.259 | 0.278 / 0.297 | 0.354 / 0.348 |
| ECL | 0.181 / 0.270 | 0.188 / 0.274 | 0.204 / 0.293 | 0.246 / 0.324 |
| Traffic (not scheduled) | 0.462 / 0.295 | 0.466 / 0.296 | 0.482 / 0.304 | 0.514 / 0.322 |
| Solar-Energy (not scheduled) | 0.234 / 0.286 | 0.267 / 0.310 | 0.290 / 0.315 | 0.289 / 0.317 |

Table 10's PatchTST numbers were obtained by the iTransformer authors (TimesNet
benchmark / their unified framework) at L=96, not necessarily with the official
PatchTST script hyperparameters above. A gap vs Table 10, if it appears, should
be read the same way as DLinear: official-script HPs vs Table 10 HPs.

## Runtime and memory (single RTX 4090 24 GB, ~60 GB RAM)

Channel-independence reshapes each batch to `(batch_size × channels)` univariate
series of 12 patches, so ECL (32 × 321 = 10,272) is the scheduled bottleneck.
Traffic (24 × 862 = 20,688), if added later, would be slower still. Model
weights themselves are tiny (≤ 1.5M). L=96 uses 12 patches vs 42 at official
L=336, so this run is lighter on VRAM than the official scripts.

`test()` accumulates `preds`, `trues` and `inputx` on the CPU (same pattern as
DLinear / iTransformer). RAM estimates in `jobs.csv` (`ram_est_gb`) are that
peak plus 4 GB overhead:

* ETTh / ETTm / Weather: about 4–8 GB. Two jobs can run together.
* ECL H96 / H192 / H336: 9.1 / 12.7 / 18.0 GB. Can share the GPU with a small job.
* ECL H720: **30.3 GB**. The scheduler will not co-schedule two H720 jobs under
  a 60 GB budget. It can run beside an ETTh job (~4 GB).
* Traffic (not scheduled): roughly 13 / 20 / 29 / **47 GB** at H96 / 192 / 336 / 720.
  H720 would have to run alone.

VRAM at L=96 should fit a single ECL job comfortably in 24 GB; two ECL trainings
are expected to fit but have not been measured on the 4090. If a job OOMs, drop
`--max-parallel` to 1. Default launch is `--max-parallel 2 --ram-budget-gb 60`.

Wall-clock (reasoning plus a CPU-only parameter / forward check on the box; no
GPU training was run). Order in `jobs.csv`: ETTh1, ETTh2, ETTm1, ETTm2,
Weather, ECL.

* ETTh1 + ETTh2 (24 jobs): small model, but patience 100 ⇒ ~100 epochs. Each
  epoch is cheap (~66 train steps). Estimate **2–4 hours** at 2-way parallelism.
* ETTm1 + ETTm2 (24 jobs): larger model, more steps, patience 20. Estimate
  **4–8 hours** if they stop around 20–40 epochs; longer if they run near 100.
* Weather (12 jobs): similar model, 21 channels, patience 20. Estimate **3–6 hours**.
* ECL (12 jobs): dominates. Patience 10, 569 train steps/epoch plus a val and
  test pass every epoch. Estimate **6–18 hours** depending on early stopping
  (roughly 30–90 min/job typical, up to ~2–3 h/job if 100 epochs). Two jobs
  may run at once except two H720s.

**Total for the 72 scheduled jobs: about 15–35 hours typical, up to ~1.5–2 days
if early stopping rarely triggers.** Dropping Exchange saves only about an hour;
ECL is still the bottleneck. The VM rental ends around 2026-11-22, so this is
tight if DLinear is still running on 20 Nov and ECL does not early-stop, but
more realistic than the earlier estimate that included Traffic.

A known official-script quirk (not changed): `--model_id $model_id_name_$seq_len`
is parsed by bash as `${model_id_name_}${seq_len}` (unset prefix), so a raw
official script actually sets `--model_id 336_96`. This reproduction writes the
intended name after the lookback override (`ETTh1_96_96`, …). `model_id` only
names checkpoints and the setting string; metrics are unaffected.

## Data

Same dataset bundle as the iTransformer / DLinear reproductions (Google Drive id
`1l51QsKvQPcqILT3DwfjCgx8Dsg2rpjot`). Expected layout under `--data-root`:
`ETT-small/`, `electricity/`, `traffic/`, `weather/`, `exchange_rate/` (and
`Solar/`, unused here). Checksums: `data/SHA256SUMS`. Data files are not committed.

Official scripts use a flat `./dataset/*.csv` layout; the scheduler maps each job
to the corresponding subdirectory of this bundle.

## Environment

Python 3.11, torch 2.5.1+cu121 (same VM runtime as iTransformer / DLinear),
numpy / pandas / scikit-learn / matplotlib, one RTX 4090 (24 GB). Official
`requirements.txt` pins `torch==1.11.0`. GPU runs are not bit-deterministic
(the official code does not enable cuDNN determinism).

## How to run

```bash
python tools/build_configs.py                      # regenerate configs/jobs.csv (LOOKBACK=96, no Traffic, no Exchange)
python tools/scheduler.py --data-root /path/to/dataset --work-dir /path/to/work --dry-run
python tools/scheduler.py --data-root /path/to/dataset --work-dir /path/to/work \
          --max-parallel 2 --ram-budget-gb 60           # skip-completed, resumable
python tools/aggregate.py --results /path/to/work/results/results.csv --readme ../README.md
```

The scheduler gives each job its own working directory, reads test MSE/MAE from
the official log line `mse:..., mae:..., rse:...`, appends one row per seed to
`results/results.csv`, writes `results/status.json`, and then deletes `pred.npy`,
`test_results/` PDFs and checkpoints. At most 2 jobs run together, and only
while the summed RAM estimate fits the budget.

Do not start this while the DLinear reproduction is still using the GPU.

## Results

The 72 runs finished on 2026-11-20 10:24 to 2026-11-21 16:49 MYT (~30 h 25 m wall clock on one RTX 4090), with 0 failures. Numbers below are the 3-seed mean ± sample standard deviation (seeds 2026/2027/2028), never the best seed. "Paper" is the PatchTST column of iTransformer Table 10 (arXiv:2310.06625v4), which PAC-Former Table 4 copies at lookback 96. Traffic and Exchange were not scheduled. Δ = (ours − paper) / paper; a negative value means a lower error than the paper.

Across the 24 dataset–horizon settings, mean MSE is 2.73% lower than the paper and mean MAE is 1.92% lower. The ECL gap (mean ΔMSE −7.15%) reflects the official script's 100-epoch / patience-10 budget versus Table 10's unified Time-Series-Library (TSLib) setting.

| Dataset | Horizon | Seeds | MSE (mean ± std) | MAE (mean ± std) | Paper MSE | Paper MAE | ΔMSE vs paper | ΔMAE vs paper |
|---|---|---|---|---|---|---|---|---|
| ETTh1 | 96 | 3 | 0.3929 ± 0.0010 | 0.4080 ± 0.0010 | 0.414 | 0.419 | -5.09% | -2.63% |
| ETTh1 | 192 | 3 | 0.4457 ± 0.0006 | 0.4350 ± 0.0001 | 0.460 | 0.445 | -3.11% | -2.25% |
| ETTh1 | 336 | 3 | 0.4836 ± 0.0012 | 0.4518 ± 0.0007 | 0.501 | 0.466 | -3.46% | -3.04% |
| ETTh1 | 720 | 3 | 0.4769 ± 0.0016 | 0.4691 ± 0.0014 | 0.500 | 0.488 | -4.63% | -3.87% |
| ETTh2 | 96 | 3 | 0.2954 ± 0.0003 | 0.3441 ± 0.0002 | 0.302 | 0.348 | -2.18% | -1.11% |
| ETTh2 | 192 | 3 | 0.3784 ± 0.0002 | 0.3941 ± 0.0002 | 0.388 | 0.400 | -2.48% | -1.48% |
| ETTh2 | 336 | 3 | 0.3816 ± 0.0007 | 0.4098 ± 0.0005 | 0.426 | 0.433 | -10.42% | -5.36% |
| ETTh2 | 720 | 3 | 0.4119 ± 0.0011 | 0.4337 ± 0.0007 | 0.431 | 0.446 | -4.44% | -2.75% |
| ETTm1 | 96 | 3 | 0.3209 ± 0.0012 | 0.3594 ± 0.0009 | 0.329 | 0.367 | -2.46% | -2.08% |
| ETTm1 | 192 | 3 | 0.3617 ± 0.0016 | 0.3816 ± 0.0008 | 0.367 | 0.385 | -1.44% | -0.90% |
| ETTm1 | 336 | 3 | 0.3924 ± 0.0031 | 0.4031 ± 0.0012 | 0.399 | 0.410 | -1.66% | -1.69% |
| ETTm1 | 720 | 3 | 0.4526 ± 0.0022 | 0.4374 ± 0.0016 | 0.454 | 0.439 | -0.30% | -0.36% |
| ETTm2 | 96 | 3 | 0.1781 ± 0.0010 | 0.2602 ± 0.0015 | 0.175 | 0.259 | +1.78% | +0.46% |
| ETTm2 | 192 | 3 | 0.2440 ± 0.0028 | 0.3034 ± 0.0022 | 0.241 | 0.302 | +1.24% | +0.47% |
| ETTm2 | 336 | 3 | 0.3064 ± 0.0046 | 0.3432 ± 0.0027 | 0.305 | 0.343 | +0.46% | +0.07% |
| ETTm2 | 720 | 3 | 0.4106 ± 0.0057 | 0.4037 ± 0.0028 | 0.402 | 0.400 | +2.14% | +0.92% |
| Weather | 96 | 3 | 0.1775 ± 0.0011 | 0.2188 ± 0.0010 | 0.177 | 0.218 | +0.30% | +0.35% |
| Weather | 192 | 3 | 0.2246 ± 0.0004 | 0.2588 ± 0.0002 | 0.225 | 0.259 | -0.17% | -0.08% |
| Weather | 336 | 3 | 0.2778 ± 0.0005 | 0.2979 ± 0.0002 | 0.278 | 0.297 | -0.07% | +0.32% |
| Weather | 720 | 3 | 0.3507 ± 0.0004 | 0.3464 ± 0.0002 | 0.354 | 0.348 | -0.93% | -0.46% |
| ECL | 96 | 3 | 0.1664 ± 0.0001 | 0.2523 ± 0.0001 | 0.181 | 0.270 | -8.07% | -6.57% |
| ECL | 192 | 3 | 0.1741 ± 0.0002 | 0.2606 ± 0.0003 | 0.188 | 0.274 | -7.40% | -4.90% |
| ECL | 336 | 3 | 0.1897 ± 0.0001 | 0.2769 ± 0.0002 | 0.204 | 0.293 | -7.02% | -5.50% |
| ECL | 720 | 3 | 0.2310 ± 0.0007 | 0.3121 ± 0.0009 | 0.246 | 0.324 | -6.11% | -3.66% |

Average relative difference over the four horizons in each dataset:

| Dataset | ΔMSE vs paper | ΔMAE vs paper |
|---|---|---|
| ETTh1 | -4.07% | -2.95% |
| ETTh2 | -4.88% | -2.68% |
| ETTm1 | -1.47% | -1.26% |
| ETTm2 | +1.40% | +0.48% |
| Weather | -0.22% | +0.03% |
| ECL | -7.15% | -5.16% |
| **Overall** | **-2.73%** | **-1.92%** |

## Files

* `official/` official supervised code (Apache-2.0, see `THIRD_PARTY_NOTICE.md`).
  `Formers/` and the self-supervised tree were not copied.
* `patches/official_changes.diff` documents that no official-code changes were made
* `configs/jobs.csv` 72-job list; `configs/official_script_args.json` parsed scripts,
  defaults, and the unscheduled Exchange proposal / Traffic note
* `configs/paper_reference.csv` iTransformer Table 10 PatchTST column (L=96)
* `tools/` config builder, scheduler, aggregation, launch script
* `results/` per-seed `results.csv` and `summary.csv`
