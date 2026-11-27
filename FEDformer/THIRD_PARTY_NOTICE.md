# Third-party notice

`official/` is a copy of **MAZiqing/FEDformer** (https://github.com/MAZiqing/FEDformer)
at commit `c0f6b972def125691434d62be1ecadf710ae921a` (2022-07-05), distributed under the
MIT License (see `official/LICENSE`).

Changes relative to that commit:

* `run.py`: the hard-coded seed (`fix_seed = 2021`) is replaced by a `--seed` argument
  (default 2021, identical behaviour when omitted). Full diff:
  `patches/official_changes.diff`.
* `utils/tools.py`: one `np.Inf` -> `np.inf`. Upstream was written against
  NumPy 1.x; `np.Inf` was removed in NumPy 2.0. Compatibility only, no logic change.
* the `dataset/` folder shipped in the upstream repository was not copied; the
  benchmark files are the ones listed in `data/SHA256SUMS`.

Nothing in the model, layers, data loaders, training loop, learning-rate schedule,
early stopping, or metrics was changed.

Paper: Zhou et al., "FEDformer: Frequency Enhanced Decomposed Transformer for
Long-term Series Forecasting", ICML 2022, arXiv:2201.12740.
