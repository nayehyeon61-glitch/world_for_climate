# E2 memory model against GraphCast and ClimODE (2020)

Latitude-weighted RMSE in physical units, means over the 31 initialisations of the GraphCast protocol in 2020 (held out for every model of ours; every 12 days, 00 UTC). GraphCast = the WeatherBench-2 forecast coarsened to 5.625 deg. E2 = N1 + 16-channel memory + energy-neutral SFNO corrector (stage 13), warm = the memory filled by 10 or 30 days of re-observed one-step forecasts before the free run. ClimODE, FourCastNet, ClimaX and IFS are the published numbers of ClimODE's Table 6 (Verma, Heinonen & Garg, ICLR 2024; WeatherBench 5.625 deg, test 2017-2018), so they come from other years and are reported to 36 h only.

| variable | lead | GraphCast | E2 warm 30 d | E2 warm 10 d | E2 cold | N1 cold | persistence | climatology | ClimODE (pub.) | E2 30 d / GraphCast | E2 30 d / ClimODE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| z500 | 6 h | 12.73 | 120 | 120 | 125 | 93.99 | 223 | 821 | 103 | 9.47x | 1.17x |
| z500 | 12 h | 23.54 | 169 | 168 | 218 | 144 | 353 | 823 | 135 | 7.17x | 1.25x |
| z500 | 18 h | 26.37 | 234 | 233 | 327 | 198 | 492 | 822 | 163 | 8.88x | 1.44x |
| z500 | 24 h | 34.78 | 290 | 290 | 426 | 258 | 579 | 824 | 193 | 8.32x | 1.50x |
| z500 | 36 h | 49.28 | 370 | 372 | 631 | 358 | 722 | 818 | 260 | 7.51x | 1.43x |
| z500 | 72 h | 116 | 630 | 642 | 1326 | 682 | 932 | 798 | - | 5.44x | - |
| z500 | 120 h | 259 | 1005 | 1074 | 2529 | 1109 | 1023 | 789 | - | 3.88x | - |
| t850 | 6 h | 0.1013 | 0.9195 | 0.9187 | 1.039 | 0.9092 | 1.179 | 3.29 | 1.16 | 9.08x | 0.79x |
| t850 | 12 h | 0.2581 | 1.44 | 1.439 | 1.66 | 1.41 | 1.984 | 3.298 | 1.32 | 5.58x | 1.09x |
| t850 | 18 h | 0.2677 | 1.656 | 1.653 | 2.067 | 1.619 | 2.38 | 3.285 | 1.47 | 6.19x | 1.13x |
| t850 | 24 h | 0.356 | 1.902 | 1.897 | 2.528 | 1.885 | 2.756 | 3.301 | 1.55 | 5.34x | 1.23x |
| t850 | 36 h | 0.4471 | 2.425 | 2.419 | 3.499 | 2.485 | 3.466 | 3.313 | 1.75 | 5.43x | 1.39x |
| t850 | 72 h | 0.7287 | 3.423 | 3.425 | 5.86 | 3.781 | 4.012 | 3.244 | - | 4.70x | - |
| t850 | 120 h | 1.304 | 4.434 | 4.539 | 8.695 | 5.097 | 4.267 | 3.203 | - | 3.40x | - |
| t2m | 6 h | 0.2687 | 1.958 | 1.958 | 2.101 | 1.662 | 2.536 | 2.454 | 1.21 | 7.29x | 1.62x |
| t2m | 12 h | 0.2924 | 2.278 | 2.278 | 2.537 | 2.063 | 3.52 | 2.479 | 1.45 | 7.79x | 1.57x |
| t2m | 18 h | 0.3233 | 2.63 | 2.626 | 2.996 | 2.32 | 2.457 | 2.465 | 1.43 | 8.13x | 1.84x |
| t2m | 24 h | 0.3525 | 2.758 | 2.748 | 3.351 | 2.424 | 1.743 | 2.457 | 1.4 | 7.82x | 1.97x |
| t2m | 36 h | 0.4183 | 3.396 | 3.374 | 4.34 | 3.123 | 3.92 | 2.5 | 1.7 | 8.12x | 2.00x |
| t2m | 72 h | 0.6018 | 4.867 | 4.795 | 7.376 | 4.571 | 2.544 | 2.413 | - | 8.09x | - |
| t2m | 120 h | 0.985 | 6.452 | 6.451 | 11.74 | 6.241 | 2.781 | 2.334 | - | 6.55x | - |
| u10 | 6 h | 0.1095 | 1.519 | 1.519 | 1.527 | 1.472 | 1.473 | 3.453 | 1.41 | 13.88x | 1.08x |
| u10 | 12 h | 0.2922 | 1.801 | 1.798 | 1.827 | 1.743 | 2.372 | 3.439 | 1.81 | 6.16x | 1.00x |
| u10 | 18 h | 0.3045 | 1.975 | 1.969 | 2 | 1.907 | 2.939 | 3.431 | 1.97 | 6.49x | 1.00x |
| u10 | 24 h | 0.4003 | 2.133 | 2.126 | 2.187 | 2.077 | 3.286 | 3.408 | 2.01 | 5.33x | 1.06x |
| u10 | 36 h | 0.5113 | 2.468 | 2.458 | 2.569 | 2.426 | 3.794 | 3.402 | 2.25 | 4.83x | 1.10x |
| u10 | 72 h | 0.9241 | 3.165 | 3.155 | 3.509 | 3.271 | 4.322 | 3.366 | - | 3.43x | - |
| u10 | 120 h | 1.719 | 3.853 | 3.844 | 4.322 | 4.17 | 4.568 | 3.417 | - | 2.24x | - |
| v10 | 6 h | 0.1119 | 1.57 | 1.569 | 1.587 | 1.533 | 1.789 | 3.645 | 1.53 | 14.04x | 1.03x |
| v10 | 12 h | 0.3076 | 1.872 | 1.869 | 1.901 | 1.838 | 2.931 | 3.615 | 1.81 | 6.09x | 1.03x |
| v10 | 18 h | 0.3232 | 2.076 | 2.071 | 2.126 | 2.037 | 3.656 | 3.633 | 1.96 | 6.42x | 1.06x |
| v10 | 24 h | 0.4258 | 2.239 | 2.233 | 2.305 | 2.209 | 4.088 | 3.603 | 2.04 | 5.26x | 1.10x |
| v10 | 36 h | 0.5479 | 2.54 | 2.534 | 2.718 | 2.545 | 4.569 | 3.584 | 2.29 | 4.64x | 1.11x |
| v10 | 72 h | 1.006 | 3.272 | 3.263 | 3.903 | 3.375 | 4.885 | 3.513 | - | 3.25x | - |
| v10 | 120 h | 1.848 | 4.005 | 3.994 | 4.698 | 4.238 | 5.003 | 3.535 | - | 2.17x | - |

## Reading

- **GraphCast:** E2 (30-day fill) is 8.3x GraphCast's z500 RMSE and 5.3x its t850 RMSE at 24 h, narrowing to 3.9x and 3.4x at 120 h (N1: 4.3x and 3.9x). The gap narrows with lead, and the memory narrows it further than N1 does from 72 h on.
- **ClimODE (published, other years):** ClimODE is better at every lead to 36 h: z500 24 h 193 vs 290, t850 1.55 vs 1.90 K, t2m 1.40 vs 2.76 K, u10 2.01 vs 2.13 m/s. The winds are closest (u10/v10 within 6-10%); the 2 m temperature is furthest (2x).
- **Weak spot:** t2m is worse than persistence from 18 h on (24 h 2.76 vs 1.74 K): the surface diurnal cycle and surface definition (stage 9 column) remain the main deficit; t850 at 120 h is at persistence (4.43 vs 4.27 K) while z500 120 h is just below it (1005 vs 1023).
- **Caveats:** one year (2020), 31 initialisations, no intervals (the protocol keeps no per-initialisation record); ClimODE's numbers are from 2017-2018 and from its own pipeline.
