# Upstream

Phase 0 does not reimplement SCP, MSE_lb, or LUR.

| Item | Value |
|---|---|
| Repository | https://github.com/WanjinVon/TS_Predictability |
| Commit | `3bd624602c95506b6f9beedfc43c041fdb6afb36` |
| Commit message | Change paper year to ICML 2026 in README |
| Paper | Beyond Model Ranking: Predictability-Aligned Evaluation for Time Series Forecasting (ICML 2026) |
| How this package gets the code | `setup.sh` clones that commit into `third_party/TS_Predictability/` (gitignored, not vendored) |

`setup.sh` checks out that commit even if `main` moves later.

## Functions this driver calls

SCP and the linear MSE lower bound:

- `Exp_Long_Term_Forecast.evaluate_predictability` in `iTransformer/experiments/exp_long_term_forecasting.py`
- `Exp_Main.evaluate_predictability` in `DLinear/exp/exp_main.py`

Frequency-band LUR (the official scripts also call this from `run_LUR_bands.py`):

- `evaluate_band_LUR` in the same two files
- Entry points: `iTransformer/run_LUR_bands.py`, `DLinear/run_LUR_bands.py`

Training entry points, MSE only (`--use_ps_loss 0`), horizon 96 unless `--horizon` is set:

- `iTransformer/run.py` with the hyperparameters from `iTransformer/scripts/ECL.sh` and `Weather.sh`
- `DLinear/run_longExp.py` with the hyperparameters from `DLinear/scripts/ECL.sh` and `weather.sh`

The upstream eval shells call `run_predictability_eval.py`, which is not in the tree (the file that exists is `run_predictability_test.py`). This driver does not use those shells. It trains with `run.py` / `run_longExp.py` and then calls `evaluate_predictability` and `evaluate_band_LUR` with the same checkpoint setting string those trainers write.

Traffic is not in the upstream diagnostic scripts. The iTransformer traffic widths and depth are the ones already used in this reproduction repo (`iTransformer/official/scripts/multivariate_forecasting/Traffic/iTransformer.sh`: 862 channels, 4 layers, `d_model` 512, `d_ff` 512, batch 16, learning rate 0.001). DLinear traffic uses the upstream ECL optimization settings (learning rate 0.001, batch 16, `seq_len = pred_len`) and `enc_in` 862. That is not the original DLinear paper lookback of 336.

## Names

In the official CSV written by `evaluate_predictability`:

- SCP in this package is the column `P_lin_mean_overall` (spectral coherence predictability). `NMSE_lb = 1 - P_lin` inside the official code.
- MSE_lb is the column `mse_lb_mean_overall`.

In the official CSV written by `evaluate_band_LUR`:

- Band LUR is the column `XR_mean` (the code's explained-power ratio). The README of the upstream repo calls this quantity LUR.
- The energy weight is `weight_mean`.

This package averages those official columns across the pre-registered channels. It does not recompute Welch spectra.

## License

As of commit `3bd624602c95506b6f9beedfc43c041fdb6afb36` the upstream repository has no root `LICENSE` file. Nested license files shipped inside that tree (for example under `PatchTST/Formers/`) stay in the clone. This reproduction does not add a license on top of that code and does not vendor the tree into git.
