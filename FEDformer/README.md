# FEDformer reproduction

Faithful re-run of **FEDformer** (Zhou et al., ICML 2022, arXiv:2201.12740) with the
authors' official code and official scripts, 3 seeds (2026, 2027, 2028).

## What is reproduced

* Multivariate long-term benchmark: ETTh1, ETTh2, ETTm1, ETTm2, Exchange, Weather,
  ECL (Electricity), Traffic; lookback 96, horizons 96 / 192 / 336 / 720.
  8 datasets x 4 horizons x 3 seeds = **96 runs** (`configs/jobs.csv`).
* `official/scripts/run_M.sh` also drives Autoformer, Informer and Transformer; only the
  `FEDformer` calls are kept here. PEMS and illness are not part of this benchmark.

## Protocol (= official code, unchanged)

* Code: `official/` = MAZiqing/FEDformer @ `c0f6b972def125691434d62be1ecadf710ae921a`.
  Changes: a `--seed` argument replacing the hard-coded `fix_seed = 2021`, and one
  `np.Inf` -> `np.inf` (NumPy 2 removed the alias). See `patches/official_changes.diff`.
* Hyperparameters: taken verbatim from `official/scripts/run_M.sh`; anything a call does
  not set falls back to the `run.py` default (`*_src` columns in `configs/jobs.csv`).
  No hyperparameter search.
* `--itr` is forced to 1: the script uses itr=3, but the repository convention is three
  explicit seeds, so the runs have to be separate.
* Fixed by the official code: Adam, MSE loss, `train_epochs` 10, early stopping on
  validation MSE with patience 3, official lr schedule, dropout, `timeF` time covariates,
  RevIN-style normalisation, test loader batch_size=1 (every test window is scored),
  metrics computed on the normalised scale.
* Seeds: 2026, 2027, 2028 (`--seed`). Main numbers are **mean +- sample std over the 3
  seeds**, never the best seed.

## Hyperparameters (from `official/scripts/run_M.sh`; seed-independent)

| Dataset | Horizon | seq_len | label_len | e_layers | d_layers | d_model | d_ff | n_heads | batch | lr |
|---|---|---|---|---|---|---|---|---|---|---|
| ETTh1 | 96 | 96 | 48 | 2 | 1 | 512 | 2048 | 8 | 32 | 0.0001 |
| ETTh1 | 192 | 96 | 48 | 2 | 1 | 512 | 2048 | 8 | 32 | 0.0001 |
| ETTh1 | 336 | 96 | 48 | 2 | 1 | 512 | 2048 | 8 | 32 | 0.0001 |
| ETTh1 | 720 | 96 | 48 | 2 | 1 | 512 | 2048 | 8 | 32 | 0.0001 |
| ETTh2 | 96 | 96 | 48 | 2 | 1 | 512 | 2048 | 8 | 32 | 0.0001 |
| ETTh2 | 192 | 96 | 48 | 2 | 1 | 512 | 2048 | 8 | 32 | 0.0001 |
| ETTh2 | 336 | 96 | 48 | 2 | 1 | 512 | 2048 | 8 | 32 | 0.0001 |
| ETTh2 | 720 | 96 | 48 | 2 | 1 | 512 | 2048 | 8 | 32 | 0.0001 |
| ETTm1 | 96 | 96 | 48 | 2 | 1 | 512 | 2048 | 8 | 32 | 0.0001 |
| ETTm1 | 192 | 96 | 48 | 2 | 1 | 512 | 2048 | 8 | 32 | 0.0001 |
| ETTm1 | 336 | 96 | 48 | 2 | 1 | 512 | 2048 | 8 | 32 | 0.0001 |
| ETTm1 | 720 | 96 | 48 | 2 | 1 | 512 | 2048 | 8 | 32 | 0.0001 |
| ETTm2 | 96 | 96 | 48 | 2 | 1 | 512 | 2048 | 8 | 32 | 0.0001 |
| ETTm2 | 192 | 96 | 48 | 2 | 1 | 512 | 2048 | 8 | 32 | 0.0001 |
| ETTm2 | 336 | 96 | 48 | 2 | 1 | 512 | 2048 | 8 | 32 | 0.0001 |
| ETTm2 | 720 | 96 | 48 | 2 | 1 | 512 | 2048 | 8 | 32 | 0.0001 |
| Exchange | 96 | 96 | 48 | 2 | 1 | 512 | 2048 | 8 | 32 | 0.0001 |
| Exchange | 192 | 96 | 48 | 2 | 1 | 512 | 2048 | 8 | 32 | 0.0001 |
| Exchange | 336 | 96 | 48 | 2 | 1 | 512 | 2048 | 8 | 32 | 0.0001 |
| Exchange | 720 | 96 | 48 | 2 | 1 | 512 | 2048 | 8 | 32 | 0.0001 |
| Weather | 96 | 96 | 48 | 2 | 1 | 512 | 2048 | 8 | 32 | 0.0001 |
| Weather | 192 | 96 | 48 | 2 | 1 | 512 | 2048 | 8 | 32 | 0.0001 |
| Weather | 336 | 96 | 48 | 2 | 1 | 512 | 2048 | 8 | 32 | 0.0001 |
| Weather | 720 | 96 | 48 | 2 | 1 | 512 | 2048 | 8 | 32 | 0.0001 |
| ECL | 96 | 96 | 48 | 2 | 1 | 512 | 2048 | 8 | 32 | 0.0001 |
| ECL | 192 | 96 | 48 | 2 | 1 | 512 | 2048 | 8 | 32 | 0.0001 |
| ECL | 336 | 96 | 48 | 2 | 1 | 512 | 2048 | 8 | 32 | 0.0001 |
| ECL | 720 | 96 | 48 | 2 | 1 | 512 | 2048 | 8 | 32 | 0.0001 |
| Traffic | 96 | 96 | 48 | 2 | 1 | 512 | 2048 | 8 | 32 | 0.0001 |
| Traffic | 192 | 96 | 48 | 2 | 1 | 512 | 2048 | 8 | 32 | 0.0001 |
| Traffic | 336 | 96 | 48 | 2 | 1 | 512 | 2048 | 8 | 32 | 0.0001 |
| Traffic | 720 | 96 | 48 | 2 | 1 | 512 | 2048 | 8 | 32 | 0.0001 |

## Results

<!-- RESULTS:BEGIN -->
_Run `tools/aggregate.py --readme README.md` after the runs finish._
<!-- RESULTS:END -->

## Known gaps

* **Solar-Energy and PEMS are not part of FEDformer's own benchmark.** The official
  `run_M.sh` covers ETTm1/ETTh1/ETTm2/ETTh2, electricity, exchange, traffic, weather and
  illness (of which illness is not in this repository's set). Running Solar-Energy
  would mean inventing a setting the authors never used, so the cell is left absent
  rather than approximated.
* `configs/paper_reference.csv` has the right schema and rows but the paper columns are
  empty: the numbers must be transcribed from arXiv:2201.12740 by hand (the arXiv HTML
  does not expose the table cleanly). `tools/aggregate.py` leaves the comparison columns
  empty until they are filled.
* The upstream repository also ships a `dataset/` folder; it was not copied, the
  benchmark files are identified by `data/SHA256SUMS` instead.

