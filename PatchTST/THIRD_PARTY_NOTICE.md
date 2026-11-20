# Third-party notice

`official/` is a copy of the **PatchTST_supervised/** tree from
**yuqinie98/PatchTST** (https://github.com/yuqinie98/PatchTST)
at commit `204c21efe0b39603ad6e2ca640ef5896646ab1a9` (2023-08-11),
distributed under the Apache License 2.0 (see `official/LICENSE`).

`PatchTST_self_supervised/` and `pic/` from that commit were not copied.
`PatchTST_supervised/Formers/` was also omitted, to keep the repository small;
those baselines are not used by this PatchTST reproduction.

The Apache-2.0 LICENSE file lives at the repository root of yuqinie98/PatchTST
and is copied here as `official/LICENSE`. `official/.commit` records the commit
hash; it is not part of the upstream tree.

Changes relative to that commit:

* None. The official `run_longExp.py` already exposes `--random_seed`
  (default 2021). This reproduction uses that flag as-is. See
  `patches/official_changes.diff` (documentation only; no code hunks).

Nothing in the model, layers, data loaders, training loop, learning-rate
schedule, early stopping, or metrics was changed.

Paper: Nie, Nguyen, Sinthong, Kalagnanam, "A Time Series is Worth 64 Words:
Long-term Forecasting with Transformers", ICLR 2023, arXiv:2211.14730.
