# Third-party notice

`official/` is a copy of **thuml/iTransformer** (https://github.com/thuml/iTransformer)
at commit `c2426e68ca13f74aaec08045c5c724d8ad328124` (2025-07-17), distributed under the
MIT License (see `official/LICENSE`, Copyright (c) 2022 THUML @ Tsinghua University).

Changes relative to that commit:

* `run.py`: the hard-coded seed (`fix_seed = 2023`) is replaced by a `--seed` argument
  (default 2023, identical behaviour when omitted). Full diff: `patches/official_changes.diff`.
* `figures/` (README images only) was not copied to keep the repository small.

Nothing in the model, layers, data loaders, training loop, learning-rate schedule,
early stopping, or metrics was changed.

Paper: Liu et al., "iTransformer: Inverted Transformers Are Effective for Time Series
Forecasting", ICLR 2024, arXiv:2310.06625.
