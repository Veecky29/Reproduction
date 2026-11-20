# Prepared root README insert for PatchTST

Do not apply until launch (after DLinear finishes and the GPU is free).
This file is a draft only; the live `/workspace/reproduction_repo/README.md`
has not been edited.

The PC clone at `C:\Users\User\Desktop\reproduction` may already have a newer
root README than the box copy (iTransformer / DLinear statuses). Insert into
whatever is current; do not overwrite existing result tables.

Also add this gitignore exception if it is not already present:

    !PatchTST/results/**/*.csv

## 1. Status table row

Add after the DLinear row:

```markdown
| PatchTST (ICLR 2023) | [`PatchTST/`](PatchTST/) | running |
```

## 2. Results section

Append after the DLinear results block:

```markdown
## PatchTST: results

Lookback 96. MSE / MAE on the standardised test set, mean ± sample std over seeds 2026/2027/2028.
"Paper" = PAC-Former Table 4 = iTransformer Table 10 (arXiv:2310.06625v4), PatchTST column.
Traffic and Exchange are not in this run. Protocol: [`PatchTST/README.md`](PatchTST/README.md).

<!-- PATCHTST_RESULTS:BEGIN -->
*Running: 72 jobs (6 datasets × 4 horizons × 3 seeds). Traffic and Exchange are not scheduled.*
<!-- PATCHTST_RESULTS:END -->
```
