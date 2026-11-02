# Stage 10: the memory model against GraphCast, N1 and ClimODE

Latitude-weighted RMSE in physical units, means over the 31 initialisations of the GraphCast protocol (every 12 days, 00 UTC), 6 h steps. GraphCast is the WeatherBench-2 forecast coarsened to the 5.625 deg grid. The memory model is D-noactor (stage 10; N1 + memory kernel, fine-tuned on 2018-2019); cold = empty memory, warm = 10 days of fully re-observed one-step forecasts first.

## 2020 (held out for every model)

| variable | lead | GraphCast | N1 cold | memory cold | memory warm | persistence | climatology |
|---|---|---|---|---|---|---|---|
| z500 | 6 h | 12.73 | 93.99 | 125 | 120 | 223 | 821 |
| z500 | 24 h | 34.78 | 258 | 426 | 290 | 579 | 824 |
| z500 | 72 h | 116 | 682 | 1326 | 642 | 932 | 798 |
| z500 | 120 h | 259 | 1109 | 2529 | 1074 | 1023 | 789 |
| t850 | 6 h | 0.1013 | 0.9092 | 1.039 | 0.9187 | 1.179 | 3.29 |
| t850 | 24 h | 0.356 | 1.885 | 2.528 | 1.897 | 2.756 | 3.301 |
| t850 | 72 h | 0.7287 | 3.781 | 5.86 | 3.425 | 4.012 | 3.244 |
| t850 | 120 h | 1.304 | 5.097 | 8.695 | 4.539 | 4.267 | 3.203 |
| q700 | 6 h | 5.411e-05 | 0.0004161 | 0.0004172 | 0.0004176 | 0.0005041 | 0.001304 |
| q700 | 24 h | 0.0002666 | 0.0008394 | 0.0008285 | 0.0008327 | 0.001183 | 0.00131 |
| q700 | 72 h | 0.0004987 | 0.001587 | 0.001575 | 0.001552 | 0.001631 | 0.001321 |
| q700 | 120 h | 0.0007327 | 0.002026 | 0.002138 | 0.001989 | 0.001709 | 0.001309 |
| u850 | 6 h | 0.1486 | 1.801 | 1.865 | 1.841 | 2.002 | 5.125 |
| u850 | 24 h | 0.636 | 2.812 | 2.993 | 2.87 | 4.658 | 5.106 |
| u850 | 72 h | 1.356 | 4.708 | 5.34 | 4.551 | 6.35 | 5.059 |
| u850 | 120 h | 2.455 | 6.152 | 7.236 | 5.685 | 6.67 | 5.08 |
| t2m | 6 h | 0.2687 | 1.662 | 2.101 | 1.958 | 2.536 | 2.454 |
| t2m | 24 h | 0.3525 | 2.424 | 3.351 | 2.748 | 1.743 | 2.457 |
| t2m | 72 h | 0.6018 | 4.571 | 7.376 | 4.795 | 2.544 | 2.413 |
| t2m | 120 h | 0.985 | 6.241 | 11.74 | 6.451 | 2.781 | 2.334 |
| msl | 6 h | 15.55 | 152 | 163 | 181 | 247 | 723 |
| msl | 24 h | 41.39 | 314 | 359 | 353 | 572 | 722 |
| msl | 72 h | 120 | 636 | 1000 | 647 | 852 | 700 |
| msl | 120 h | 258 | 931 | 2062 | 1012 | 910 | 693 |
| u10 | 6 h | 0.1095 | 1.472 | 1.527 | 1.519 | 1.473 | 3.453 |
| u10 | 24 h | 0.4003 | 2.077 | 2.187 | 2.126 | 3.286 | 3.408 |
| u10 | 72 h | 0.9241 | 3.271 | 3.509 | 3.155 | 4.322 | 3.366 |
| u10 | 120 h | 1.719 | 4.17 | 4.322 | 3.844 | 4.568 | 3.417 |
| v10 | 6 h | 0.1119 | 1.533 | 1.587 | 1.569 | 1.789 | 3.645 |
| v10 | 24 h | 0.4258 | 2.209 | 2.305 | 2.233 | 4.088 | 3.603 |
| v10 | 72 h | 1.006 | 3.375 | 3.903 | 3.263 | 4.885 | 3.513 |
| v10 | 120 h | 1.848 | 4.238 | 4.698 | 3.994 | 5.003 | 3.535 |
| tp6 | 6 h | 0.0002435 | 0.0008475 | 0.0008849 | 0.0008872 | 0.0008637 | 0.001236 |
| tp6 | 24 h | 0.0004632 | 0.001049 | 0.001139 | 0.00112 | 0.001602 | 0.001316 |
| tp6 | 72 h | 0.0007284 | 0.001323 | 0.001357 | 0.001299 | 0.001754 | 0.001279 |
| tp6 | 120 h | 0.0009233 | 0.001446 | 0.001477 | 0.001386 | 0.001774 | 0.001263 |

ACC:

| variable | lead | GraphCast | N1 cold | memory cold | memory warm | persistence |
|---|---|---|---|---|---|---|
| z500 | 24 h | 0.9991 | 0.9506 | 0.8837 | 0.9363 | 0.7495 |
| z500 | 72 h | 0.9893 | 0.6816 | 0.4004 | 0.6816 | 0.3356 |
| z500 | 120 h | 0.9437 | 0.3141 | 0.07156 | 0.3239 | 0.1959 |
| t850 | 24 h | 0.9941 | 0.8457 | 0.7654 | 0.8382 | 0.6507 |
| t850 | 72 h | 0.9742 | 0.5139 | 0.3209 | 0.5567 | 0.2495 |
| t850 | 120 h | 0.913 | 0.2353 | 0.04617 | 0.3102 | 0.1438 |
| t2m | 24 h | 0.9894 | 0.6615 | 0.5377 | 0.6017 | 0.7405 |
| t2m | 72 h | 0.9672 | 0.3373 | 0.1555 | 0.3299 | 0.44 |
| t2m | 120 h | 0.9049 | 0.1296 | -0.05712 | 0.1301 | 0.3187 |
| u10 | 24 h | 0.993 | 0.8004 | 0.773 | 0.7845 | 0.5367 |
| u10 | 72 h | 0.9612 | 0.4978 | 0.4063 | 0.4842 | 0.1939 |
| u10 | 120 h | 0.8646 | 0.2357 | 0.1303 | 0.2567 | 0.11 |

Blow-ups (samples that kept their last finite forecast): warm 0 of 31


## Cause ablations (2020)

The warm start has not finished.

## Against GraphCast (2020)

| variable | N1 / GraphCast 24 h | N1 / GraphCast 72 h | N1 / GraphCast 120 h | memory cold / GraphCast 24 h | memory cold / GraphCast 72 h | memory cold / GraphCast 120 h | memory warm / GraphCast 24 h | memory warm / GraphCast 72 h | memory warm / GraphCast 120 h |
|---|---|---|---|---|---|---|---|---|---|
| z500 | 7.42x | 5.90x | 4.29x | 12.25x | 11.46x | 9.77x | 8.33x | 5.55x | 4.15x |
| t850 | 5.29x | 5.19x | 3.91x | 7.10x | 8.04x | 6.67x | 5.33x | 4.70x | 3.48x |
| t2m | 6.87x | 7.60x | 6.34x | 9.51x | 12.26x | 11.92x | 7.80x | 7.97x | 6.55x |
| u10 | 5.19x | 3.54x | 2.43x | 5.46x | 3.80x | 2.51x | 5.31x | 3.41x | 2.24x |


## Against ClimODE (2018; ClimODE's Table 6 is its 2017-2018 test)

ClimODE, FourCastNet, ClimaX and IFS as printed in Verma et al. (2024); ours on the 31 initialisations of 2018. 2018 is a fine-tuning year of the memory model (its N1 base was trained on 2015-2017), so its 2018 scores are a reference, not a held-out test.

No 2018 scores yet.

## Fill length: 10 days, 30 days, two years (2020)

Per variable: RMSE of the free run after a 10-day spin-up, a 30-day spin-up and a continuous fill from 2018-01-01 (memory carried through two years); gains against the 10-day spin-up, + = better.

| variable | lead | 10 d | 30 d | 2 y | gain 30 d | gain 2 y | 2 y vs 30 d |
|---|---|---|---|---|---|---|---|
| z500 | 24 h | 290 | 290 | 313 | +0.1% | -8.1% | -8.1% |
| z500 | 72 h | 642 | 630 | 663 | +1.9% | -3.3% | -5.3% |
| z500 | 120 h | 1074 | 1005 | 1042 | +6.4% | +2.9% | -3.7% |
| t850 | 24 h | 1.897 | 1.902 | 1.964 | -0.2% | -3.5% | -3.3% |
| t850 | 72 h | 3.425 | 3.423 | 3.492 | +0.1% | -2.0% | -2.0% |
| t850 | 120 h | 4.539 | 4.434 | 4.483 | +2.3% | +1.2% | -1.1% |
| q700 | 24 h | 0.0008327 | 0.0008329 | 0.0008357 | -0.0% | -0.4% | -0.3% |
| q700 | 72 h | 0.001552 | 0.001553 | 0.001559 | -0.0% | -0.4% | -0.4% |
| q700 | 120 h | 0.001989 | 0.001989 | 0.00199 | -0.0% | -0.1% | -0.1% |
| u850 | 24 h | 2.87 | 2.884 | 2.904 | -0.5% | -1.2% | -0.7% |
| u850 | 72 h | 4.551 | 4.566 | 4.585 | -0.3% | -0.7% | -0.4% |
| u850 | 120 h | 5.685 | 5.681 | 5.683 | +0.1% | +0.0% | -0.0% |
| t2m | 24 h | 2.748 | 2.758 | 2.83 | -0.4% | -3.0% | -2.6% |
| t2m | 72 h | 4.795 | 4.867 | 5.027 | -1.5% | -4.8% | -3.3% |
| t2m | 120 h | 6.451 | 6.452 | 6.635 | -0.0% | -2.9% | -2.8% |
| msl | 24 h | 353 | 357 | 366 | -1.2% | -3.9% | -2.6% |
| msl | 72 h | 647 | 642 | 658 | +0.7% | -1.7% | -2.4% |
| msl | 120 h | 1012 | 971 | 988 | +4.1% | +2.4% | -1.8% |
| u10 | 24 h | 2.126 | 2.133 | 2.145 | -0.4% | -0.9% | -0.6% |
| u10 | 72 h | 3.155 | 3.165 | 3.177 | -0.3% | -0.7% | -0.4% |
| u10 | 120 h | 3.844 | 3.853 | 3.859 | -0.2% | -0.4% | -0.2% |
| v10 | 24 h | 2.233 | 2.239 | 2.243 | -0.3% | -0.5% | -0.2% |
| v10 | 72 h | 3.263 | 3.272 | 3.283 | -0.3% | -0.6% | -0.3% |
| v10 | 120 h | 3.994 | 4.005 | 4.015 | -0.3% | -0.5% | -0.2% |
| tp6 | 24 h | 0.00112 | 0.001122 | 0.001127 | -0.2% | -0.7% | -0.5% |
| tp6 | 72 h | 0.001299 | 0.0013 | 0.001303 | -0.1% | -0.3% | -0.2% |
| tp6 | 120 h | 0.001386 | 0.001386 | 0.001386 | +0.0% | -0.0% | -0.0% |

Mean gain of the two-year fill over the 30-day spin-up (z500, t850, t2m, u10): 24 h -3.6%, 72 h -2.7%, 120 h -1.9%; blow-ups {'30 d': 0, '2 y': 0}


## Decision rules

- **C1** memory helps the forecast itself: the warm start lowers the cold start's RMSE by >= 3% at 24 h (mean over z500, t850, t2m, u10) while warm_no_memory stays within 1% of the cold start  
  → pending
- **C2** lead: the lead at which the warm-start gain falls below 1% (where the filled memory stops mattering)  
  → pending
- **C3** GraphCast: RMSE ratio to GraphCast (WB2 forecasts coarsened to 5.625 deg) at 24 h and 72 h; <= 1.5 would be GraphCast-class, which no model of this stage is expected to reach  
  → N1: 24 h 6.20x, 72 h 5.56x, 120 h 4.24x; memory cold: 24 h 8.58x, 72 h 8.89x, 120 h 7.72x; memory warm: 24 h 6.69x, 72 h 5.41x, 120 h 4.10x; 
- **C4** ClimODE (indirect): on 2018 at 24 h the cold memory model's RMSE is below ClimODE's published RMSE on >= 3 of z500, t850, t2m, u10 (2018 is a fine-tuning year of the memory model; ClimODE's test years are 2017-2018 after ten training years)  
  → pending
- **C5** observations (감자 09:49): with the points outside a hidden cap re-observed every 6 h, the memory model's RMSE on the cap is >= 10% below N1 + nudging at 24 h (mean over z500, t850, t2m, u10; the stage 10 P1 in physical units)  
  → pending
- **C6** crossing: the first lead at which the observation-fed memory model beats GraphCast's free forecast on the cap (mean ratio < 1), if any within 120 h  
  → pending
- **L1** saturation (10.2 cheap test): the two-year continuous fill and the 30-day spin-up differ by <= 1% at 24 h and 120 h (mean over z500, t850, t2m, u10) with the stage 10 kernel (tau <= 10 days)  
  → not supported (two-year fill vs 30 days: 24 h -3.6%, 120 h -1.9%, + = the longer fill is better; the memory state after two years is not the state after 30 days, so the kernel's effective time scale is longer than its nominal tau)
