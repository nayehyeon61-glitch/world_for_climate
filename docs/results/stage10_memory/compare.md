# Stage 10: the memory model against GraphCast, N1 and ClimODE

Latitude-weighted RMSE in physical units, means over the 31 initialisations of the GraphCast protocol (every 12 days, 00 UTC), 6 h steps. GraphCast is the WeatherBench-2 forecast coarsened to the 5.625 deg grid. The memory model is D-noactor (stage 10; N1 + memory kernel, fine-tuned on 2018-2019); cold = empty memory, warm = 10 days of fully re-observed one-step forecasts first.

## 2020 (held out for every model)

| variable | lead | GraphCast | N1 cold | memory cold | memory warm | warm, no memory | E2 cold | E2 warm | persistence | climatology |
|---|---|---|---|---|---|---|---|---|---|---|
| z500 | 6 h | 12.73 | 93.99 | 155 | 122 | 128 | 125 | 120 | 223 | 821 |
| z500 | 24 h | 34.78 | 258 | 601 | 301 | 547 | 426 | 290 | 579 | 824 |
| z500 | 72 h | 116 | 682 | 2241 | 766 | 2178 | 1326 | 642 | 932 | 798 |
| z500 | 120 h | 259 | 1109 | - | 1703 | 5073 | 2529 | 1074 | 1023 | 789 |
| t850 | 6 h | 0.1013 | 0.9092 | 1.201 | 0.9554 | 1.095 | 1.039 | 0.9187 | 1.179 | 3.29 |
| t850 | 24 h | 0.356 | 1.885 | 3.45 | 2.035 | 2.932 | 2.528 | 1.897 | 2.756 | 3.301 |
| t850 | 72 h | 0.7287 | 3.781 | 9.446 | 4.054 | 7.822 | 5.86 | 3.425 | 4.012 | 3.244 |
| t850 | 120 h | 1.304 | 5.097 | - | 6.9 | 14.52 | 8.695 | 4.539 | 4.267 | 3.203 |
| q700 | 6 h | 5.411e-05 | 0.0004161 | 0.0004232 | 0.0004248 | 0.000422 | 0.0004172 | 0.0004176 | 0.0005041 | 0.001304 |
| q700 | 24 h | 0.0002666 | 0.0008394 | 0.0008905 | 0.0008877 | 0.0008793 | 0.0008285 | 0.0008327 | 0.001183 | 0.00131 |
| q700 | 72 h | 0.0004987 | 0.001587 | 0.002026 | 0.001827 | 0.001817 | 0.001575 | 0.001552 | 0.001631 | 0.001321 |
| q700 | 120 h | 0.0007327 | 0.002026 | - | 0.002463 | 0.002589 | 0.002138 | 0.001989 | 0.001709 | 0.001309 |
| u850 | 6 h | 0.1486 | 1.801 | 1.914 | 1.869 | 1.91 | 1.865 | 1.841 | 2.002 | 5.125 |
| u850 | 24 h | 0.636 | 2.812 | 3.164 | 2.902 | 3.129 | 2.993 | 2.87 | 4.658 | 5.106 |
| u850 | 72 h | 1.356 | 4.708 | 6.5 | 4.732 | 6.248 | 5.34 | 4.551 | 6.35 | 5.059 |
| u850 | 120 h | 2.455 | 6.152 | - | 6.452 | 9.457 | 7.236 | 5.685 | 6.67 | 5.08 |
| t2m | 6 h | 0.2687 | 1.662 | 2.414 | 2.085 | 2.289 | 2.101 | 1.958 | 2.536 | 2.454 |
| t2m | 24 h | 0.3525 | 2.424 | 5.157 | 3.001 | 4.23 | 3.351 | 2.748 | 1.743 | 2.457 |
| t2m | 72 h | 0.6018 | 4.571 | 14.24 | 6.231 | 12.08 | 7.376 | 4.795 | 2.544 | 2.413 |
| t2m | 120 h | 0.985 | 6.241 | - | 11 | 20.97 | 11.74 | 6.451 | 2.781 | 2.334 |
| msl | 6 h | 15.55 | 152 | 174 | 185 | 176 | 163 | 181 | 247 | 723 |
| msl | 24 h | 41.39 | 314 | 413 | 358 | 420 | 359 | 353 | 572 | 722 |
| msl | 72 h | 120 | 636 | 1531 | 708 | 1604 | 1000 | 647 | 852 | 700 |
| msl | 120 h | 258 | 931 | - | 1326 | 4330 | 2062 | 1012 | 910 | 693 |
| u10 | 6 h | 0.1095 | 1.472 | 1.553 | 1.54 | 1.549 | 1.527 | 1.519 | 1.473 | 3.453 |
| u10 | 24 h | 0.4003 | 2.077 | 2.298 | 2.17 | 2.255 | 2.187 | 2.126 | 3.286 | 3.408 |
| u10 | 72 h | 0.9241 | 3.271 | 4.09 | 3.419 | 3.836 | 3.509 | 3.155 | 4.322 | 3.366 |
| u10 | 120 h | 1.719 | 4.17 | - | 4.271 | 4.71 | 4.322 | 3.844 | 4.568 | 3.417 |
| v10 | 6 h | 0.1119 | 1.533 | 1.628 | 1.608 | 1.627 | 1.587 | 1.569 | 1.789 | 3.645 |
| v10 | 24 h | 0.4258 | 2.209 | 2.406 | 2.297 | 2.382 | 2.305 | 2.233 | 4.088 | 3.603 |
| v10 | 72 h | 1.006 | 3.375 | 4.737 | 3.588 | 4.163 | 3.903 | 3.263 | 4.885 | 3.513 |
| v10 | 120 h | 1.848 | 4.238 | - | 4.534 | 4.989 | 4.698 | 3.994 | 5.003 | 3.535 |
| tp6 | 6 h | 0.0002435 | 0.0008475 | 0.0009105 | 0.0009302 | 0.0009353 | 0.0008849 | 0.0008872 | 0.0008637 | 0.001236 |
| tp6 | 24 h | 0.0004632 | 0.001049 | 0.001131 | 0.001129 | 0.001196 | 0.001139 | 0.00112 | 0.001602 | 0.001316 |
| tp6 | 72 h | 0.0007284 | 0.001323 | 0.001459 | 0.001332 | 0.001371 | 0.001357 | 0.001299 | 0.001754 | 0.001279 |
| tp6 | 120 h | 0.0009233 | 0.001446 | - | 0.001468 | 0.001506 | 0.001477 | 0.001386 | 0.001774 | 0.001263 |

ACC:

| variable | lead | GraphCast | N1 cold | memory cold | memory warm | warm, no memory | E2 cold | E2 warm | persistence |
|---|---|---|---|---|---|---|---|---|---|
| z500 | 24 h | 0.9991 | 0.9506 | 0.8068 | 0.9317 | 0.8165 | 0.8837 | 0.9363 | 0.7495 |
| z500 | 72 h | 0.9893 | 0.6816 | 0.2285 | 0.6344 | 0.1894 | 0.4004 | 0.6816 | 0.3356 |
| z500 | 120 h | 0.9437 | 0.3141 | - | 0.2207 | -0.01185 | 0.07156 | 0.3239 | 0.1959 |
| t850 | 24 h | 0.9941 | 0.8457 | 0.6762 | 0.8253 | 0.7142 | 0.7654 | 0.8382 | 0.6507 |
| t850 | 72 h | 0.9742 | 0.5139 | 0.1802 | 0.519 | 0.192 | 0.3209 | 0.5567 | 0.2495 |
| t850 | 120 h | 0.913 | 0.2353 | - | 0.2133 | -0.03299 | 0.04617 | 0.3102 | 0.1438 |
| t2m | 24 h | 0.9894 | 0.6615 | 0.4237 | 0.6038 | 0.4423 | 0.5377 | 0.6017 | 0.7405 |
| t2m | 72 h | 0.9672 | 0.3373 | 0.06625 | 0.2986 | 0.02936 | 0.1555 | 0.3299 | 0.44 |
| t2m | 120 h | 0.9049 | 0.1296 | - | 0.06001 | -0.1177 | -0.05712 | 0.1301 | 0.3187 |
| u10 | 24 h | 0.993 | 0.8004 | 0.7526 | 0.7778 | 0.7564 | 0.773 | 0.7845 | 0.5367 |
| u10 | 72 h | 0.9612 | 0.4978 | 0.2952 | 0.4528 | 0.3199 | 0.4063 | 0.4842 | 0.1939 |
| u10 | 120 h | 0.8646 | 0.2357 | - | 0.215 | 0.07306 | 0.1303 | 0.2567 | 0.11 |

Blow-ups (samples that kept their last finite forecast): warm 3, warm_no_memory 4 of 31


## Cause ablations (2020)

Mean relative RMSE change over z500, t850, t2m, u10 (negative = better):

| lead | warm vs cold (filled memory) | warm_no_memory vs cold (spin-up alone) | warm vs warm_no_memory (the kernel) |
|---|---|---|---|
| 6 h | -14.0% | -8.0% | -6.6% |
| 12 h | -23.5% | -9.4% | -16.0% |
| 18 h | -29.0% | -11.0% | -20.9% |
| 24 h | -34.6% | -11.0% | -27.1% |
| 30 h | -38.1% | -11.6% | -30.8% |
| 36 h | -40.8% | -11.4% | -33.6% |
| 42 h | -42.8% | -11.5% | -35.9% |
| 48 h | -44.9% | -11.1% | -38.4% |
| 54 h | -46.2% | -11.0% | -39.9% |
| 60 h | -47.4% | -11.0% | -41.0% |
| 66 h | -48.1% | -10.8% | -42.0% |
| 72 h | -48.9% | -10.3% | -43.1% |
| 78 h | -49.0% | -9.8% | -43.5% |
| 84 h | -49.2% | -9.7% | -43.8% |
| 90 h | -49.3% | -9.5% | -44.1% |
| 96 h | - | - | -44.4% |
| 102 h | - | - | -44.3% |
| 108 h | - | - | -44.2% |
| 114 h | - | - | -44.2% |
| 120 h | - | - | -43.9% |

Per variable, warm vs cold:

| variable | 6 h | 24 h | 72 h | 120 h |
|---|---|---|---|---|
| z500 | -21.1% | -50.0% | -65.8% | - |
| t850 | -20.4% | -41.0% | -57.1% | - |
| q700 | +0.4% | -0.3% | -9.8% | - |
| u850 | -2.3% | -8.3% | -27.2% | - |
| t2m | -13.6% | -41.8% | -56.2% | - |
| msl | +5.9% | -13.3% | -53.8% | - |
| u10 | -0.8% | -5.6% | -16.4% | - |
| v10 | -1.2% | -4.5% | -24.3% | - |
| tp6 | +2.2% | -0.2% | -8.7% | - |


## Against GraphCast (2020)

| variable | N1 / GraphCast 24 h | N1 / GraphCast 72 h | N1 / GraphCast 120 h | memory cold / GraphCast 24 h | memory cold / GraphCast 72 h | memory cold / GraphCast 120 h | memory warm / GraphCast 24 h | memory warm / GraphCast 72 h | memory warm / GraphCast 120 h |
|---|---|---|---|---|---|---|---|---|---|
| z500 | 7.42x | 5.90x | 4.29x | 17.29x | 19.36x | - | 8.64x | 6.62x | 6.58x |
| t850 | 5.29x | 5.19x | 3.91x | 9.69x | 12.96x | - | 5.72x | 5.56x | 5.29x |
| t2m | 6.87x | 7.60x | 6.34x | 14.63x | 23.66x | - | 8.51x | 10.35x | 11.17x |
| u10 | 5.19x | 3.54x | 2.43x | 5.74x | 4.43x | - | 5.42x | 3.70x | 2.48x |


## Against ClimODE (2018; ClimODE's Table 6 is its 2017-2018 test)

ClimODE, FourCastNet, ClimaX and IFS as printed in Verma et al. (2024); ours on the 31 initialisations of 2018. 2018 is a fine-tuning year of the memory model (its N1 base was trained on 2015-2017), so its 2018 scores are a reference, not a held-out test.

| variable | lead | ClimODE | FourCastNet | ClimaX | IFS | GraphCast (WB2) | N0 | N1 | memory cold |
|---|---|---|---|---|---|---|---|---|---|
| z500 | 6 h | 103 | 149 | 248 | 26.9 | 12.48 | 122 | 93.38 | 154 |
| z500 | 12 h | 135 | 218 | 265 | - | 23.83 | 159 | 142 | 284 |
| z500 | 18 h | 163 | 275 | 320 | - | 25.91 | 201 | 193 | 441 |
| z500 | 24 h | 193 | 333 | 365 | 51 | 34.11 | 256 | 254 | 592 |
| z500 | 36 h | 260 | 449 | 455 | - | 48.31 | 363 | 354 | 915 |
| t850 | 6 h | 1.16 | 1.18 | 1.64 | 0.69 | 0.09601 | 0.9815 | 0.9121 | 1.189 |
| t850 | 12 h | 1.32 | 1.47 | 1.77 | - | 0.2558 | 1.487 | 1.403 | 2.003 |
| t850 | 18 h | 1.47 | 1.65 | 1.93 | - | 0.2647 | 1.588 | 1.604 | 2.676 |
| t850 | 24 h | 1.55 | 1.83 | 2.17 | 0.87 | 0.3457 | 1.794 | 1.852 | 3.377 |
| t850 | 36 h | 1.75 | 2.21 | 2.49 | - | 0.4288 | 2.521 | 2.445 | 4.861 |
| t2m | 6 h | 1.21 | 1.28 | 2.02 | 0.97 | 0.2612 | 2.713 | 1.662 | 2.384 |
| t2m | 12 h | 1.45 | 1.48 | 2.26 | - | 0.2921 | 3.996 | 2.073 | 3.235 |
| t2m | 18 h | 1.43 | 1.61 | 2.45 | - | 0.3239 | 3.393 | 2.342 | 4.192 |
| t2m | 24 h | 1.4 | 1.68 | 2.37 | 1.02 | 0.3457 | 3.119 | 2.415 | 5.101 |
| t2m | 36 h | 1.7 | 1.9 | 2.87 | - | 0.4099 | 5.764 | 3.085 | 7.301 |
| u10 | 6 h | 1.41 | 1.47 | 1.58 | 0.8 | 0.1109 | 3.937 | 1.456 | 1.535 |
| u10 | 12 h | 1.81 | 1.89 | 1.96 | - | 0.2956 | 3.842 | 1.712 | 1.839 |
| u10 | 18 h | 1.97 | 2.05 | 2.24 | - | 0.3069 | 4.172 | 1.873 | 2.049 |
| u10 | 24 h | 2.01 | 2.33 | 2.49 | 1.11 | 0.4019 | 4.248 | 2.061 | 2.302 |
| u10 | 36 h | 2.25 | 2.87 | 2.98 | - | 0.5108 | 4.73 | 2.404 | 2.775 |
| v10 | 6 h | 1.53 | 1.54 | 1.6 | 0.94 | 0.1128 | 3.598 | 1.517 | 1.609 |
| v10 | 12 h | 1.81 | 1.81 | 1.97 | - | 0.3109 | 3.345 | 1.814 | 1.898 |
| v10 | 18 h | 1.96 | 2.11 | 2.26 | - | 0.3263 | 3.641 | 1.994 | 2.125 |
| v10 | 24 h | 2.04 | 2.39 | 2.48 | 1.33 | 0.4305 | 3.829 | 2.159 | 2.347 |
| v10 | 36 h | 2.29 | 2.95 | 2.98 | - | 0.5495 | 4.326 | 2.486 | 2.905 |


## Forecasts fed by observations outside a hidden cap (2020)

Hidden cap of 25% of the sphere per initialisation (mean hidden weight 0.25); RMSE over the cap only. Blow-ups (of 31): warm 31, nudging 1.

| variable | lead | GraphCast (free) | N1 + nudging | memory (posterior) | memory / GraphCast | memory / nudging |
|---|---|---|---|---|---|---|
| z500 | 6 h | 12.71 | 91.95 | 115 | 9.07x | 1.25x |
| t850 | 6 h | 0.1041 | 0.8896 | 0.9243 | 8.88x | 1.04x |
| t850 | 24 h | 0.3559 | 1.814 | blown | - | - |
| t850 | 72 h | 0.7354 | 3.385 | blown | - | - |
| t850 | 120 h | 1.234 | 4.134 | blown | - | - |
| q700 | 6 h | 5.418e-05 | 0.0004268 | 0.0004353 | 8.04x | 1.02x |
| q700 | 24 h | 0.0002585 | 0.0008577 | blown | - | - |
| q700 | 72 h | 0.0005047 | 0.001497 | blown | - | - |
| q700 | 120 h | 0.0007565 | 0.001916 | blown | - | - |
| u850 | 6 h | 0.1466 | 1.686 | 1.719 | 11.72x | 1.02x |
| u850 | 24 h | 0.6236 | 2.71 | blown | - | - |
| u850 | 72 h | 1.368 | 4.309 | blown | - | - |
| u850 | 120 h | 2.322 | 5.362 | blown | - | - |
| t2m | 6 h | 0.2356 | 1.496 | 1.86 | 7.90x | 1.24x |
| t2m | 24 h | 0.3219 | 2.399 | blown | - | - |
| t2m | 72 h | 0.5439 | 5.207 | blown | - | - |
| t2m | 120 h | 0.851 | 7.298 | blown | - | - |
| msl | 6 h | 15.35 | 145 | 179 | 11.66x | 1.23x |
| msl | 24 h | 41.27 | 344 | blown | - | - |
| msl | 72 h | 118 | 802 | blown | - | - |
| msl | 120 h | 230 | 1217 | blown | - | - |
| u10 | 6 h | 0.1074 | 1.469 | 1.529 | 14.24x | 1.04x |
| u10 | 24 h | 0.4084 | 2.735 | blown | - | - |
| u10 | 72 h | 0.9547 | 3.945 | blown | - | - |
| u10 | 120 h | 1.674 | 4.512 | blown | - | - |
| v10 | 6 h | 0.1125 | 1.531 | 1.598 | 14.20x | 1.04x |
| v10 | 24 h | 0.4303 | 2.84 | blown | - | - |
| v10 | 72 h | 1.04 | 3.981 | blown | - | - |
| v10 | 120 h | 1.824 | 4.466 | blown | - | - |
| tp6 | 6 h | 0.0002564 | 0.0008828 | 0.000973 | 3.79x | 1.10x |
| tp6 | 24 h | 0.0004431 | 0.001048 | 0.002123 | 4.79x | 2.03x |
| tp6 | 72 h | 0.000719 | 0.001289 | 0.002412 | 3.35x | 1.87x |
| tp6 | 120 h | 0.0009066 | 0.001423 | 0.002398 | 2.64x | 1.68x |

Further variants of the same protocol (RMSE on the cap as a ratio to N1 + nudging; blown = the mean is dominated by blown-up samples):

| variant | blow-ups | z500 6 h | z500 24 h | z500 72 h | z500 120 h | t850 6 h | t850 24 h | t850 72 h | t850 120 h | t2m 6 h | t2m 24 h | t2m 72 h | t2m 120 h | u10 6 h | u10 24 h | u10 72 h | u10 120 h |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E1 bound | 31 | 1.27x | 6.93x | blown | blown | 1.03x | 8.98x | blown | blown | 1.23x | 7.54x | blown | blown | 1.04x | 2.28x | blown | blown |
| no readout | 3 | 1.25x | 1.09x | 0.96x | 0.97x | 1.04x | 1.09x | 1.01x | 1.01x | 1.24x | 1.78x | 1.89x | 1.86x | 1.04x | 1.05x | 1.05x | 1.04x |
| no kernel | 2 | 1.28x | 1.71x | 2.21x | 2.77x | 1.20x | 1.48x | 1.78x | 2.25x | 1.41x | 2.44x | 2.88x | 2.74x | 1.05x | 1.07x | 1.09x | 1.08x |
| readout max 0.5 | 31 | 1.25x | 4.51x | blown | blown | 1.04x | 6.37x | blown | blown | 1.24x | 6.24x | blown | blown | 1.04x | 2.21x | blown | blown |
| readout max 1 | 31 | 1.25x | 6.26x | 3.83x | 2.89x | 1.04x | 8.18x | 5.10x | 4.23x | 1.24x | 7.38x | 3.95x | 2.83x | 1.04x | 2.36x | 1.70x | 1.53x |
| readout max 2 | 31 | 1.25x | blown | blown | blown | 1.04x | blown | blown | blown | 1.24x | blown | blown | blown | 1.04x | blown | blown | blown |
| readout decay 1 | 28 | 1.25x | 3.81x | 2.80x | 2.41x | 1.04x | 4.70x | 2.78x | 2.46x | 1.24x | 4.58x | 2.88x | 2.13x | 1.04x | 1.70x | 1.32x | 1.20x |
| readout decay 3 | 31 | 1.25x | 6.86x | blown | blown | 1.04x | 7.22x | blown | blown | 1.24x | 6.62x | blown | blown | 1.04x | 2.45x | blown | blown |
| E2 corrector | 31 | 1.24x | 19.67x | 10.35x | 7.61x | 1.00x | 9.36x | 5.77x | 4.77x | 1.14x | blown | blown | blown | 1.02x | blown | blown | blown |


## Fill length: 10 days, 30 days, two years (2020)

Per variable: RMSE of the free run after a 10-day spin-up, a 30-day spin-up and a continuous fill from 2018-01-01 (memory carried through two years); gains against the 10-day spin-up, + = better.

| variable | lead | 10 d | 30 d | 2 y | gain 30 d | gain 2 y | 2 y vs 30 d |
|---|---|---|---|---|---|---|---|
| z500 | 24 h | 301 | 294 | 326 | +2.0% | -8.4% | -10.7% |
| z500 | 72 h | 766 | 733 | 728 | +4.3% | +4.9% | +0.7% |
| z500 | 120 h | 1703 | 1576 | 1469 | +7.5% | +13.7% | +6.8% |
| t850 | 24 h | 2.035 | 2.028 | 2.123 | +0.4% | -4.3% | -4.7% |
| t850 | 72 h | 4.054 | 3.998 | 3.884 | +1.4% | +4.2% | +2.9% |
| t850 | 120 h | 6.9 | 6.642 | 6.173 | +3.7% | +10.5% | +7.1% |
| q700 | 24 h | 0.0008877 | 0.0008865 | 0.0008908 | +0.1% | -0.4% | -0.5% |
| q700 | 72 h | 0.001827 | 0.001822 | 0.00182 | +0.3% | +0.4% | +0.1% |
| q700 | 120 h | 0.002463 | 0.00246 | 0.002428 | +0.1% | +1.4% | +1.3% |
| u850 | 24 h | 2.902 | 2.905 | 2.913 | -0.1% | -0.4% | -0.3% |
| u850 | 72 h | 4.732 | 4.728 | 4.716 | +0.1% | +0.3% | +0.3% |
| u850 | 120 h | 6.452 | 6.385 | 6.258 | +1.0% | +3.0% | +2.0% |
| t2m | 24 h | 3.001 | 2.989 | 2.945 | +0.4% | +1.9% | +1.5% |
| t2m | 72 h | 6.231 | 6.112 | 5.765 | +1.9% | +7.5% | +5.7% |
| t2m | 120 h | 11 | 10.56 | 9.772 | +4.0% | +11.2% | +7.5% |
| msl | 24 h | 358 | 356 | 364 | +0.5% | -1.5% | -2.1% |
| msl | 72 h | 708 | 692 | 692 | +2.3% | +2.2% | -0.1% |
| msl | 120 h | 1326 | 1241 | 1184 | +6.4% | +10.7% | +4.6% |
| u10 | 24 h | 2.17 | 2.17 | 2.178 | -0.0% | -0.4% | -0.4% |
| u10 | 72 h | 3.419 | 3.419 | 3.417 | -0.0% | +0.1% | +0.1% |
| u10 | 120 h | 4.271 | 4.265 | 4.234 | +0.1% | +0.9% | +0.7% |
| v10 | 24 h | 2.297 | 2.297 | 2.305 | +0.0% | -0.3% | -0.3% |
| v10 | 72 h | 3.588 | 3.583 | 3.583 | +0.2% | +0.2% | -0.0% |
| v10 | 120 h | 4.534 | 4.516 | 4.456 | +0.4% | +1.7% | +1.3% |
| tp6 | 24 h | 0.001129 | 0.001131 | 0.001134 | -0.2% | -0.4% | -0.2% |
| tp6 | 72 h | 0.001332 | 0.00133 | 0.001334 | +0.1% | -0.2% | -0.3% |
| tp6 | 120 h | 0.001468 | 0.001462 | 0.001451 | +0.4% | +1.2% | +0.7% |

Mean gain of the two-year fill over the 30-day spin-up (z500, t850, t2m, u10): 24 h -3.6%, 72 h +2.3%, 120 h +5.5%; blow-ups {'30 d': 3, '2 y': 3}


## Decision rules

- **C1** memory helps the forecast itself: the warm start lowers the cold start's RMSE by >= 3% at 24 h (mean over z500, t850, t2m, u10) while warm_no_memory stays within 1% of the cold start  
  → not supported (warm vs cold -34.6%, spin-up alone -11.0%)
- **C2** lead: the lead at which the warm-start gain falls below 1% (where the filled memory stops mattering)  
  → the gain stays above 1% to 120 h
- **C3** GraphCast: RMSE ratio to GraphCast (WB2 forecasts coarsened to 5.625 deg) at 24 h and 72 h; <= 1.5 would be GraphCast-class, which no model of this stage is expected to reach  
  → N1: 24 h 6.20x, 72 h 5.56x, 120 h 4.24x; memory cold: 24 h 11.84x, 72 h 15.10x, 120 h -x; memory warm: 24 h 7.07x, 72 h 6.56x, 120 h 6.38x; 
- **C4** ClimODE (indirect): on 2018 at 24 h the cold memory model's RMSE is below ClimODE's published RMSE on >= 3 of z500, t850, t2m, u10 (2018 is a fine-tuning year of the memory model; ClimODE's test years are 2017-2018 after ten training years)  
  → not supported (memory cold below ClimODE at 24 h on 0 of 5 variables)
- **C5** observations (감자 09:49): with the points outside a hidden cap re-observed every 6 h, the memory model's RMSE on the cap is >= 10% below N1 + nudging at 24 h (mean over z500, t850, t2m, u10; the stage 10 P1 in physical units)  
  → not supported (the memory model blew up before 24 h on most initialisations; 6 h ratio 1.14x; blow-ups {'warm': 31, 'nudging': 1})
- **C6** crossing: the first lead at which the observation-fed memory model beats GraphCast's free forecast on the cap (mean ratio < 1), if any within 120 h  
  → no crossing within the forecast (memory / GraphCast on the cap: 6 h 10.02x; blow-ups {'warm': 31, 'nudging': 1})
- **L1** saturation (10.2 cheap test): the two-year continuous fill and the 30-day spin-up differ by <= 1% at 24 h and 120 h (mean over z500, t850, t2m, u10) with the stage 10 kernel (tau <= 10 days)  
  → not supported (two-year fill vs 30 days: 24 h -3.6%, 120 h +5.5%, + = the longer fill is better; the memory state after two years is not the state after 30 days, so the kernel's effective time scale is longer than its nominal tau)
