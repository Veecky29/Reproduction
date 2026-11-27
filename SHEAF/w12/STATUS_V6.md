# v6 (shared encoder, stats in self branch, diag masked, alpha=0.1)

_updated 2026-11-27 10:37:04_
- 17/18, fails=0

| dataset | arm | seed | ep | test_mse | alpha | gate_mean |
|---|---|---|---|---|---|---|
| Electricity | concat | 2026 | 16 | 0.1765 | - | - |
| Electricity | concat | 2027 | 19 | 0.1745 | - | - |
| Electricity | concat | 2028 | 19 | 0.1737 | - | - |
| Electricity | plain | 2026 | 8 | 0.1939 | - | - |
| Electricity | plain | 2027 | 6 | 0.1928 | - | - |
| Electricity | plain | 2028 | 7 | 0.1924 | - | - |
| Electricity | sheaf | 2026 | 6 | 0.1920 | 0.07441 | 0.08619 |
| Electricity | sheaf | 2027 | 7 | 0.1920 | 0.07357 | 0.09777 |
| Electricity | sheaf | 2028 | 9 | 0.1916 | 0.07085 | 0.12782 |
| Weather | concat | 2026 | 3 | 0.1854 | - | - |
| Weather | concat | 2027 | 14 | 0.1871 | - | - |
| Weather | concat | 2028 | 5 | 0.2033 | - | - |
| Weather | plain | 2026 | 4 | 0.1949 | - | - |
| Weather | plain | 2027 | 3 | 0.1963 | - | - |
| Weather | plain | 2028 | 3 | 0.1893 | - | - |
| Weather | sheaf | 2026 | 2 | 0.1928 | 0.10674 | 0.49493 |
| Weather | sheaf | 2027 | 5 | 0.1915 | 0.12121 | 0.42313 |
| Weather | sheaf | 2028 | 4 | 0.1912 | 0.12535 | 0.47128 |