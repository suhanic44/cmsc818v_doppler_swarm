# Final experimental checks

Read-only checks of the main research evaluation.

## 1. Six-UAV coordinated-baseline errors

Attempts: **600**; errors: **35**; timeouts: **0**.

### Exact recorded error messages

| Count | Error message |
|---:|---|
| 35 | `Process exited without result: 1` |

### Error distribution

| Speed (m/s) | Category | Errors |
|---:|---|---:|
| 1.0 | direct_threat | 12 |
| 3.0 | direct_threat | 11 |
| 5.0 | direct_threat | 12 |

### Example encounter IDs

- `s8675309_n6_direct_threat_015_v5`: `Process exited without result: 1`
- `s8675309_n6_direct_threat_016_v1`: `Process exited without result: 1`
- `s8675309_n6_direct_threat_016_v3`: `Process exited without result: 1`
- `s8675309_n6_direct_threat_016_v5`: `Process exited without result: 1`
- `s8675309_n6_direct_threat_017_v1`: `Process exited without result: 1`
- `s8675309_n6_direct_threat_017_v3`: `Process exited without result: 1`
- `s8675309_n6_direct_threat_017_v5`: `Process exited without result: 1`
- `s8675309_n6_direct_threat_018_v1`: `Process exited without result: 1`
- `s8675309_n6_direct_threat_018_v3`: `Process exited without result: 1`
- `s8675309_n6_direct_threat_018_v5`: `Process exited without result: 1`

**Interpretation:** These are recorded worker errors, not observed collisions. Use the exact messages to decide whether follow-up is warranted. This report cannot diagnose unlogged stack traces.

## 2. Paired movement comparison audit

Metrics records: **18000**; paired rows: **14400**; expected pair keys: **14400**.

Duplicate metrics keys: **0**; duplicate pair keys: **0**; unexpected pair keys: **0**.

Audit issues (up to 30 shown): **0**.

**PASS:** Saved pair flags and movement differences agree with raw metrics.

### Safety-matched movement: common-safe encounters only

The percent reduction compares mean movement across the same common-safe encounters: 100 × (1 − mean A / mean B). It is not the average of per-trial reductions.

| A | B | UAVs | Subset | Common safe | Mean A (m) | Mean B (m) | Mean A − B (m) | Reduction vs B | A moved less |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| estimate | oracle | 4 | all | 369 | 0.1194 | 0.0858 | +0.0336 | -39.2% | 14.1% |
| estimate | oracle | 4 | threats | 219 | 0.1938 | 0.1441 | +0.0497 | -34.5% | 22.8% |
| estimate | oracle | 6 | all | 548 | 0.1232 | 0.1033 | +0.0198 | -19.2% | 23.0% |
| estimate | oracle | 6 | threats | 398 | 0.1685 | 0.1423 | +0.0263 | -18.5% | 31.7% |
| estimate | oracle | 8 | all | 580 | 0.1119 | 0.1058 | +0.0061 | -5.8% | 31.0% |
| estimate | oracle | 8 | threats | 430 | 0.1506 | 0.1426 | +0.0079 | -5.6% | 41.6% |
| estimate | oracle | 10 | all | 596 | 0.1096 | 0.1074 | +0.0022 | -2.0% | 32.0% |
| estimate | oracle | 10 | threats | 446 | 0.1457 | 0.1435 | +0.0023 | -1.6% | 42.8% |
| estimate | oracle | 16 | all | 599 | 0.1073 | 0.1096 | -0.0023 | 2.1% | 35.7% |
| estimate | oracle | 16 | threats | 449 | 0.1426 | 0.1461 | -0.0035 | 2.4% | 47.7% |
| estimate | oracle | 25 | all | 600 | 0.1044 | 0.1089 | -0.0045 | 4.1% | 37.0% |
| estimate | oracle | 25 | threats | 450 | 0.1392 | 0.1452 | -0.0060 | 4.1% | 49.3% |
| estimate | oracle | all | all | 3292 | 0.1120 | 0.1047 | +0.0073 | -7.0% | 29.9% |
| estimate | oracle | all | threats | 2392 | 0.1530 | 0.1440 | +0.0090 | -6.2% | 41.1% |
| estimate | dopplerdodge | 4 | all | 348 | 0.1156 | 0.7862 | -0.6707 | 85.3% | 88.5% |
| estimate | dopplerdodge | 4 | threats | 204 | 0.1897 | 0.9596 | -0.7699 | 80.2% | 100.0% |
| estimate | dopplerdodge | 6 | all | 470 | 0.1213 | 1.0455 | -0.9242 | 88.4% | 93.8% |
| estimate | dopplerdodge | 6 | threats | 333 | 0.1701 | 1.1473 | -0.9772 | 85.2% | 99.4% |
| estimate | dopplerdodge | 8 | all | 537 | 0.1099 | 1.3031 | -1.1931 | 91.6% | 96.8% |
| estimate | dopplerdodge | 8 | threats | 398 | 0.1479 | 1.4133 | -1.2654 | 89.5% | 100.0% |
| estimate | dopplerdodge | 10 | all | 536 | 0.1081 | 1.5540 | -1.4459 | 93.0% | 98.7% |
| estimate | dopplerdodge | 10 | threats | 400 | 0.1446 | 1.6245 | -1.4799 | 91.1% | 100.0% |
| estimate | dopplerdodge | 16 | all | 533 | 0.1084 | 2.1546 | -2.0462 | 95.0% | 99.4% |
| estimate | dopplerdodge | 16 | threats | 399 | 0.1442 | 2.2590 | -2.1148 | 93.6% | 100.0% |
| estimate | dopplerdodge | 25 | all | 494 | 0.1039 | 3.0808 | -2.9768 | 96.6% | 99.6% |
| estimate | dopplerdodge | 25 | threats | 369 | 0.1391 | 3.1817 | -3.0426 | 95.6% | 100.0% |
| estimate | dopplerdodge | all | all | 2918 | 0.1108 | 1.7025 | -1.5917 | 93.5% | 96.6% |
| estimate | dopplerdodge | all | threats | 2103 | 0.1526 | 1.8381 | -1.6855 | 91.7% | 99.9% |
| estimate | dopplerdodge_coordinated | 4 | all | 350 | 0.1151 | 0.7443 | -0.6292 | 84.5% | 88.3% |
| estimate | dopplerdodge_coordinated | 4 | threats | 206 | 0.1877 | 0.9076 | -0.7199 | 79.3% | 99.5% |
| estimate | dopplerdodge_coordinated | 6 | all | 482 | 0.1230 | 1.0157 | -0.8927 | 87.9% | 94.0% |
| estimate | dopplerdodge_coordinated | 6 | threats | 344 | 0.1712 | 1.1119 | -0.9406 | 84.6% | 99.4% |
| estimate | dopplerdodge_coordinated | 8 | all | 550 | 0.1115 | 1.2564 | -1.1449 | 91.1% | 96.9% |
| estimate | dopplerdodge_coordinated | 8 | threats | 411 | 0.1488 | 1.3556 | -1.2068 | 89.0% | 100.0% |
| estimate | dopplerdodge_coordinated | 10 | all | 546 | 0.1085 | 1.4936 | -1.3850 | 92.7% | 98.7% |
| estimate | dopplerdodge_coordinated | 10 | threats | 408 | 0.1450 | 1.5591 | -1.4141 | 90.7% | 100.0% |
| estimate | dopplerdodge_coordinated | 16 | all | 551 | 0.1071 | 2.0871 | -1.9800 | 94.9% | 99.5% |
| estimate | dopplerdodge_coordinated | 16 | threats | 413 | 0.1424 | 2.1928 | -2.0504 | 93.5% | 100.0% |
| estimate | dopplerdodge_coordinated | 25 | all | 347 | 0.0974 | 2.9823 | -2.8849 | 96.7% | 99.4% |
| estimate | dopplerdodge_coordinated | 25 | threats | 248 | 0.1363 | 3.1172 | -2.9809 | 95.6% | 100.0% |
| estimate | dopplerdodge_coordinated | all | all | 2826 | 0.1108 | 1.5716 | -1.4609 | 93.0% | 96.5% |
| estimate | dopplerdodge_coordinated | all | threats | 2030 | 0.1530 | 1.6953 | -1.5423 | 91.0% | 99.9% |
| dopplerdodge_coordinated | dopplerdodge | 4 | all | 559 | 0.7915 | 0.8380 | -0.0466 | 5.6% | 92.8% |
| dopplerdodge_coordinated | dopplerdodge | 4 | threats | 416 | 0.8888 | 0.9411 | -0.0523 | 5.6% | 100.0% |
| dopplerdodge_coordinated | dopplerdodge | 6 | all | 509 | 1.0028 | 1.0618 | -0.0590 | 5.6% | 94.7% |
| dopplerdodge_coordinated | dopplerdodge | 6 | threats | 376 | 1.0908 | 1.1549 | -0.0642 | 5.6% | 100.0% |
| dopplerdodge_coordinated | dopplerdodge | 8 | all | 546 | 1.2314 | 1.3038 | -0.0724 | 5.6% | 96.9% |
| dopplerdodge_coordinated | dopplerdodge | 8 | threats | 411 | 1.3298 | 1.4080 | -0.0782 | 5.6% | 100.0% |
| dopplerdodge_coordinated | dopplerdodge | 10 | all | 530 | 1.4650 | 1.5512 | -0.0862 | 5.6% | 98.7% |
| dopplerdodge_coordinated | dopplerdodge | 10 | threats | 396 | 1.5326 | 1.6227 | -0.0902 | 5.6% | 100.0% |
| dopplerdodge_coordinated | dopplerdodge | 16 | all | 528 | 2.0260 | 2.1452 | -0.1192 | 5.6% | 99.4% |
| dopplerdodge_coordinated | dopplerdodge | 16 | threats | 395 | 2.1272 | 2.2523 | -0.1251 | 5.6% | 100.0% |
| dopplerdodge_coordinated | dopplerdodge | 25 | all | 333 | 2.8857 | 3.0554 | -0.1697 | 5.6% | 99.4% |
| dopplerdodge_coordinated | dopplerdodge | 25 | threats | 239 | 3.0088 | 3.1858 | -0.1770 | 5.6% | 100.0% |
| dopplerdodge_coordinated | dopplerdodge | all | all | 3005 | 1.4750 | 1.5617 | -0.0868 | 5.6% | 96.8% |
| dopplerdodge_coordinated | dopplerdodge | all | threats | 2233 | 1.5641 | 1.6561 | -0.0920 | 5.6% | 100.0% |

**Important:** These are conditional movement comparisons, not claims of equal safety. Report overall safety and timeout rates separately. 'Threats' excludes safe_miss.

