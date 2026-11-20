# Third-party notice

`official/` is a copy of **plumprc/RTSF** (https://github.com/plumprc/RTSF)
at commit `0fec00104f754f4fbf795b9b4da5fa2459b32e76` (2023-09-13).

The upstream repository at that commit does **not declare a license** (GitHub
reports `license: null`, and there is no LICENSE file in the tree). The copy
here is for research reproduction of Li, Qi, Li and Xu, "Revisiting Long-term
Time Series Forecasting: An Investigation on Linear Mapping", arXiv:2305.10721.

`pics/` and `tutorial.ipynb` from that commit were not copied, to keep the
repository small; they are not used by this RLinear reproduction.
`official/.commit` records the commit hash; it is not part of the upstream tree.

Changes relative to that commit (full diff: `patches/official_changes.diff`):

* `run.py`: the hard-coded seed (`fix_seed = 1024`, set before argparse) is
  replaced by a `--seed` argument (default 1024, identical behaviour when
  omitted).
* `exp/exp_main.py`: `--model Linear` is aliased to the `RLinear` class.
  Official `script.md` passes `--model Linear`, but the model dictionary at
  this commit only registers `RLinear`. Jobs in this reproduction emit
  `--model RLinear`; the alias is so a verbatim official command would also run.

Nothing in the model, layers, data loaders, training loop, learning-rate
schedule, early stopping, or metrics was changed.

Paper: Li, Qi, Li, Xu, "Revisiting Long-term Time Series Forecasting: An
Investigation on Linear Mapping", arXiv:2305.10721, 2023.
