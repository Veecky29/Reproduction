# Third-party notice

`official/` is a copy of **cure-lab/LTSF-Linear** (https://github.com/cure-lab/LTSF-Linear)
at commit `0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6` (2024-01-28), distributed under the
Apache License 2.0 (see `official/LICENSE`, Copyright 2022 DLinear Authors).

`FEDformer/`, `Pyraformer/` and `pics/` from that commit were not copied, to keep the
repository small; they are not used by DLinear.

Changes relative to that commit:

* `run_longExp.py`: the hard-coded seed (`fix_seed = 2021`) is replaced by a `--seed`
  argument (default 2021, identical behaviour when omitted). Full diff:
  `patches/official_changes.diff`.

Nothing in the model, layers, data loaders, training loop, learning-rate schedule,
early stopping, or metrics was changed. The decomposition kernel remains hard-coded
at 25 in `models/DLinear.py`.

Paper: Zeng, Chen, Zhang, Xu, "Are Transformers Effective for Time Series Forecasting?",
AAAI 2023, arXiv:2205.13504.
