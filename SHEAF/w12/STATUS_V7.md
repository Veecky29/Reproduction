# v7 switchable mixing

_updated 13:36:21_
- 29/30, fails=0

| dataset | arm | seed | ep | test_mse | beta_mean |
|---|---|---|---|---|---|
| ETTh1 | concat | 2026 | 2 | 0.3983 | - |
| ETTh1 | concat | 2027 | 1 | 0.4041 | - |
| ETTh1 | concat | 2028 | 2 | 0.4041 | - |
| ETTh1 | mix | 2026 | 3 | 0.3926 | 0.09521 |
| ETTh1 | mix | 2027 | 1 | 0.4161 | 0.09682 |
| ETTh1 | mix | 2028 | 3 | 0.3943 | 0.11346 |
| ETTh2 | concat | 2026 | 1 | 0.2955 | - |
| ETTh2 | concat | 2027 | 3 | 0.3097 | - |
| ETTh2 | concat | 2028 | 1 | 0.3084 | - |
| ETTh2 | mix | 2026 | 1 | 0.2970 | 0.10834 |
| ETTh2 | mix | 2027 | 4 | 0.3066 | 0.10469 |
| ETTh2 | mix | 2028 | 1 | 0.3051 | 0.10641 |
| ETTm1 | concat | 2026 | 5 | 0.3237 | - |
| ETTm1 | concat | 2027 | 8 | 0.3379 | - |
| ETTm1 | concat | 2028 | 2 | 0.3615 | - |
| ETTm1 | mix | 2026 | 1 | 0.4066 | 0.12342 |
| ETTm1 | mix | 2027 | 1 | 0.3817 | 0.11367 |
| ETTm1 | mix | 2028 | 5 | 0.3602 | 0.10921 |
| ETTm2 | concat | 2026 | 1 | 0.1770 | - |
| ETTm2 | concat | 2027 | 1 | 0.1857 | - |
| ETTm2 | concat | 2028 | 1 | 0.1853 | - |
| ETTm2 | mix | 2026 | 1 | 0.1749 | 0.11106 |
| ETTm2 | mix | 2027 | 2 | 0.1762 | 0.1066 |
| ETTm2 | mix | 2028 | 1 | 0.1774 | 0.10581 |
| Electricity | mix | 2026 | 7 | 0.1921 | 0.08765 |
| Electricity | mix | 2027 | 5 | 0.1952 | 0.10074 |
| Electricity | mix | 2028 | 7 | 0.1939 | 0.07655 |
| Weather | mix | 2026 | 4 | 0.1837 | 0.05566 |
| Weather | mix | 2027 | 5 | 0.1826 | 0.05937 |
| Weather | mix | 2028 | 5 | 0.1727 | 0.10316 |