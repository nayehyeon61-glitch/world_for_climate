# Stage 12: active observation on the memory model

The frozen stage 10 memory model (D-noactor) plays the observation game (train/active_obs.py): after each 6 h forecast a rule picks which equal-area patches to observe (budget 25% = 6 of 24), the posterior takes them in, the rest keeps the forecast plus the memory's readout. Rules: the learned policy, random (control), age (round-robin), readout (where the memory expects the largest correction), tendency (largest 6 h change), full (upper bound) and none (cold free run). Design and rules A1-A6: README_KO §12.

## Learning curve

| quarter | error (obs part) | error (free part) | entropy (first pick, nat) | blow-ups | grad |
|---|---|---|---|---|---|
| 0-24 | 0.367 | 0.962 | 3.17 | 14 | 0.176 |
| 25-49 | 0.376 | 0.805 | nan | 9 | 0.432 |
| 50-74 | 0.295 | 0.345 | 3.17 | 0 | 0.681 |
| 75-99 | 0.300 | 0.375 | 3.15 | 1 | 0.638 |

100 iterations, 102 min.

## The observation game (2020)

31 initialisations (2020-01-01 .. 2020-12-31), 120 h episodes: observations chosen after the first 8 forecasts (48 h), then free. GraphCast-weighted all-point MSE (normalised units) of the forecast before each re-observation; 'free' = mean over the free leads (54-120 h); gains (+ = better) against memory:random (paired, bootstrap 95% over initialisations). Blow-ups are counted and scored 10 from then on.

| run | budget | 6 h | 24 h | 48 h | 72 h | 120 h | free | gain vs random (+ = better) | blow-ups | selection entropy (nat) | rho(freq, log tau) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| memory:policy | 0.75 | 0.113 | 0.279 | 0.276 | 0.275 | 2.149 | 0.681 | +76.7% [+40.9%, +116.8%] | 5 | 3.16 | -0.59 |
| memory:random | 0.75 | 0.113 | 0.393 | 1.085 | 1.394 | 5.250 | 2.924 | reference | 15 | 3.18 | -0.08 |
| memory:age | 0.75 | 0.113 | 0.331 | 0.363 | 0.306 | 1.064 | 0.495 | +83.1% [+48.0%, +121.9%] | 1 | 3.18 | +0.16 |
| memory:readout | 0.75 | 0.113 | 0.330 | 5.693 | 9.188 | 10.000 | 9.375 | -220.6% [-261.2%, -172.1%] | 31 | 3.08 | -0.57 |
| memory:tendency | 0.75 | 0.113 | 0.254 | 0.270 | 0.268 | 0.989 | 0.538 | +81.6% [+42.6%, +121.9%] | 1 | 3.17 | -0.44 |
| memory:full | 0.75 | 0.113 | 0.112 | 0.115 | 0.170 | 0.403 | 0.239 | +91.8% [+57.4%, +129.8%] | 0 | - | - |
| memory:none | 0.75 | 0.115 | 0.187 | 0.363 | 0.644 | 4.296 | 1.440 | +50.7% [+14.0%, +91.4%] | 9 | - | - |
| memory:policy@0.5 | 0.5 | 0.113 | 0.712 | 2.592 | 2.701 | 6.891 | 4.093 | -40.0% [-98.3%, +20.7%] | 19 | 3.15 | -0.81 |
| memory:random@0.5 | 0.5 | 0.113 | 1.864 | 8.193 | 10.000 | 10.000 | 9.901 | -238.6% [-272.4%, -200.5%] | 31 | 3.18 | -0.18 |
| memory:age@0.5 | 0.5 | 0.113 | 0.577 | 2.083 | 2.940 | 4.312 | 3.130 | -7.0% [-65.3%, +43.8%] | 10 | 3.18 | +0.53 |
| memory:readout@0.5 | 0.5 | 0.113 | 3.975 | 9.787 | 10.000 | 10.000 | 9.991 | -241.7% [-276.0%, -203.8%] | 31 | 2.72 | -0.58 |
| memory:tendency@0.5 | 0.5 | 0.113 | 0.748 | 2.385 | 2.765 | 8.315 | 4.571 | -56.3% [-109.9%, -1.8%] | 24 | 3.16 | -0.75 |
| memory:policy@0.875 | 0.875 | 0.113 | 0.185 | 0.185 | 0.217 | 0.670 | 0.319 | +89.1% [+54.4%, +127.3%] | 0 | 3.17 | -0.63 |
| memory:random@0.875 | 0.875 | 0.113 | 0.235 | 0.259 | 0.304 | 2.760 | 1.136 | +61.2% [+30.7%, +96.1%] | 7 | 3.18 | -0.19 |
| memory:age@0.875 | 0.875 | 0.113 | 0.214 | 0.238 | 0.239 | 0.876 | 0.366 | +87.5% [+52.8%, +126.1%] | 1 | 3.18 | -0.14 |
| memory:readout@0.875 | 0.875 | 0.113 | 0.216 | 2.028 | 5.154 | 9.368 | 6.904 | -136.1% [-191.5%, -76.5%] | 28 | 3.16 | -0.61 |
| memory:tendency@0.875 | 0.875 | 0.113 | 0.186 | 0.193 | 0.219 | 0.566 | 0.313 | +89.3% [+54.8%, +127.4%] | 0 | 3.18 | -0.27 |
| n1:policy | 0.75 | 0.100 | 0.112 | 0.127 | 0.182 | 0.361 | 0.238 | +91.9% [+57.5%, +129.8%] | 0 | 3.11 | -0.52 |
| n1:random | 0.75 | 0.100 | 0.107 | 0.107 | 0.160 | 0.338 | 0.216 | +92.6% [+58.2%, +130.6%] | 0 | 3.18 | -0.08 |
| n1:full | 0.75 | 0.100 | 0.099 | 0.099 | 0.152 | 0.329 | 0.207 | +92.9% [+58.4%, +130.9%] | 0 | - | - |
| n1:none | 0.75 | 0.100 | 0.151 | 0.237 | 0.337 | 0.817 | 0.422 | +85.6% [+50.9%, +124.1%] | 1 | - | - |
| n1:policy@0.5 | 0.5 | 0.100 | 0.125 | 0.158 | 0.222 | 0.401 | 0.278 | +90.5% [+56.1%, +128.6%] | 0 | 2.94 | -0.46 |
| n1:random@0.5 | 0.5 | 0.100 | 0.120 | 0.124 | 0.176 | 0.354 | 0.232 | +92.1% [+57.5%, +130.0%] | 0 | 3.18 | -0.18 |
| n1:policy@0.875 | 0.875 | 0.100 | 0.105 | 0.111 | 0.164 | 0.341 | 0.219 | +92.5% [+58.1%, +130.5%] | 0 | 3.16 | -0.50 |
| n1:random@0.875 | 0.875 | 0.100 | 0.102 | 0.102 | 0.156 | 0.334 | 0.212 | +92.8% [+58.3%, +130.7%] | 0 | 3.18 | -0.19 |

Headline RMSE (physical units) at 24 / 72 / 120 h:

| run | z500 | t850 | t2m | u10 |
|---|---|---|---|---|
| memory:policy | 312 / 462 / 1.35e+03 | 2.61 / 3.02 / 5.98 | 3.69 / 4.51 / 8.66 | 2.67 / 2.9 / 4.13 |
| memory:random | 349 / 762 / 1.59e+03 | 2.85 / 4.63 / 6.67 | 3.92 / 6.4 / 9.26 | 3.03 / 3.42 / 4.31 |
| memory:age | 323 / 588 / 1.59e+03 | 2.6 / 3.71 / 6.7 | 3.56 / 5.12 / 10 | 2.82 / 3.09 / 4.18 |
| memory:readout | 605 / 1.03e+03 / 1.54e+03 | 4.51 / 4.77 / 6.65 | 5.58 / 6.12 / 9.52 | 3.17 / 3.76 / 4.35 |
| memory:tendency | 288 / 494 / 1.38e+03 | 2.34 / 3.21 / 6.02 | 3.48 / 4.69 / 9.23 | 2.53 / 2.91 / 4.09 |
| memory:full | 104 / 308 / 1.03e+03 | 0.957 / 1.99 / 4.59 | 2.13 / 3.23 / 7.38 | 1.49 / 2.16 / 3.5 |
| memory:none | 528 / 1.78e+03 / 3.22e+03 | 2.98 / 7.64 / 10.9 | 4.75 / 11.7 / 15.6 | 2.27 / 4.05 / 4.81 |
| memory:policy@0.5 | 701 / 975 / 1.87e+03 | 5.67 / 5.26 / 7.38 | 6.77 / 6.85 / 9.93 | 4.29 / 3.92 / 4.55 |
| memory:random@0.5 | 2.53e+10 / 1.05e+03 / 1.88e+03 | 1.12e+05 / 4.77 / 7.27 | 1.85e+12 / 5.84 / 10.2 | 2.08e+11 / 4.04 / 4.53 |
| memory:age@0.5 | 574 / 976 / 2.19e+03 | 3.95 / 4.91 / 8.35 | 4.8 / 6.18 / 12 | 3.68 / 3.75 / 4.63 |
| memory:readout@0.5 | 2.1e+03 / 1.91e+03 / 2.06e+03 | 11.4 / 6.24 / 7.49 | 13 / 8.45 / 10.8 | 5.1 / 4.91 / 4.83 |
| memory:tendency@0.5 | 644 / 1.02e+03 / 2.04e+03 | 5.48 / 5.43 / 7.29 | 6.73 / 7.11 / 9.63 | 4.51 / 4.02 / 5.62 |
| memory:policy@0.875 | 188 / 368 / 1.16e+03 | 1.63 / 2.43 / 5.12 | 2.74 / 3.85 / 8.13 | 2.08 / 2.53 / 3.93 |
| memory:random@0.875 | 233 / 489 / inf | 1.99 / 3.18 / 4.65e+15 | 3.04 / 4.63 / 1.2e+16 | 2.35 / 2.81 / inf |
| memory:age@0.875 | 217 / 442 / 1.38e+03 | 1.82 / 2.91 / 5.95 | 2.91 / 4.22 / 9.1 | 2.28 / 2.7 / 3.9 |
| memory:readout@0.875 | 382 / 876 / 1.39e+03 | 2.98 / 4.73 / 6.11 | 4.07 / 6.69 / 8.28 | 2.38 / 3.61 / 4.46 |
| memory:tendency@0.875 | 190 / 372 / 1.16e+03 | 1.6 / 2.45 / 5.15 | 2.71 / 3.86 / 8.16 | 2.17 / 2.57 / 3.78 |
| n1:policy | 133 / 339 / 755 | 1.14 / 2.23 / 3.94 | 1.88 / 3.19 / 5.06 | 1.74 / 2.46 / 3.59 |
| n1:random | 116 / 270 / 702 | 1.05 / 1.95 / 3.74 | 1.94 / 2.65 / 4.63 | 1.63 / 2.23 / 3.42 |
| n1:full | 96.7 / 251 / 683 | 0.914 / 1.85 / 3.68 | 1.72 / 2.49 / 4.56 | 1.44 / 2.06 / 3.3 |
| n1:none | 253 / 684 / 1.17e+03 | 1.85 / 3.72 / 5.33 | 2.5 / 4.63 / 6.67 | 2.06 / 3.34 / 4.21 |
| n1:policy@0.5 | 154 / 407 / 817 | 1.31 / 2.56 / 4.19 | 2 / 3.69 / 5.4 | 1.96 / 2.84 / 3.85 |
| n1:random@0.5 | 147 / 313 / 741 | 1.25 / 2.14 / 3.87 | 2.11 / 2.91 / 4.8 | 1.92 / 2.49 / 3.6 |
| n1:policy@0.875 | 114 / 281 / 707 | 1.03 / 2.01 / 3.79 | 1.81 / 2.85 / 4.81 | 1.6 / 2.26 / 3.43 |
| n1:random@0.875 | 105 / 261 / 693 | 0.975 / 1.9 / 3.7 | 1.84 / 2.58 / 4.59 | 1.53 / 2.15 / 3.36 |

## Cause ablation: the observation budget

Free-lead MSE of the policy against the random rule at the same budget (memory model; gain + = better):

| budget | random | policy | policy gain vs random | age | readout | tendency |
|---|---|---|---|---|---|---|
| 0.75 | 2.924 | 0.681 | +76.7% | 0.495 | 9.375 | 0.538 |
| 0.5 | 9.901 | 4.093 | +58.7% | 3.130 | 9.991 | 4.571 |
| 0.875 | 1.136 | 0.319 | +71.9% | 0.366 | 6.904 | 0.313 |

## Where the rules look

Selection frequency per patch (fraction of observation times; north at the top, longitude of the sector's west edge + 30):

**memory:policy**

| band \ lon | 30E | 90E | 150E | 210E | 270E | 330E |
|---|---|---|---|---|---|---|
| +49 | 0.42 | 0.49 | 0.62 | 0.55 | 0.58 | 0.67 |
| +14 | 0.76 | 0.83 | 0.77 | 0.80 | 0.80 | 0.69 |
| -14 | 0.81 | 0.75 | 0.79 | 0.68 | 0.79 | 0.68 |
| -49 | 0.96 | 0.96 | 0.91 | 0.87 | 0.90 | 0.94 |

**memory:age**

| band \ lon | 30E | 90E | 150E | 210E | 270E | 330E |
|---|---|---|---|---|---|---|
| +49 | 0.77 | 0.74 | 0.78 | 0.75 | 0.75 | 0.75 |
| +14 | 0.73 | 0.76 | 0.73 | 0.74 | 0.75 | 0.75 |
| -14 | 0.77 | 0.76 | 0.74 | 0.75 | 0.76 | 0.74 |
| -49 | 0.74 | 0.75 | 0.75 | 0.76 | 0.74 | 0.75 |

**memory:readout**

| band \ lon | 30E | 90E | 150E | 210E | 270E | 330E |
|---|---|---|---|---|---|---|
| +49 | 0.18 | 0.18 | 0.25 | 0.27 | 0.29 | 0.38 |
| +14 | 0.96 | 0.99 | 1.00 | 0.92 | 1.00 | 0.92 |
| -14 | 1.00 | 0.99 | 0.99 | 0.98 | 0.99 | 1.00 |
| -49 | 0.88 | 0.85 | 0.80 | 0.80 | 0.66 | 0.72 |

**memory:tendency**

| band \ lon | 30E | 90E | 150E | 210E | 270E | 330E |
|---|---|---|---|---|---|---|
| +49 | 0.54 | 0.69 | 0.69 | 0.65 | 0.75 | 0.78 |
| +14 | 0.81 | 0.82 | 0.80 | 0.77 | 0.81 | 0.72 |
| -14 | 0.73 | 0.77 | 0.72 | 0.68 | 0.71 | 0.64 |
| -49 | 0.87 | 0.86 | 0.77 | 0.74 | 0.83 | 0.84 |

**The memory's timescale, log10 tau (hours), patch mean under full observation**

| band \ lon | 30E | 90E | 150E | 210E | 270E | 330E |
|---|---|---|---|---|---|---|
| +49 | 1.37 | 1.37 | 1.32 | 1.33 | 1.35 | 1.33 |
| +14 | 1.34 | 1.27 | 1.27 | 1.29 | 1.30 | 1.31 |
| -14 | 1.32 | 1.28 | 1.28 | 1.30 | 1.32 | 1.32 |
| -49 | 1.29 | 1.30 | 1.31 | 1.30 | 1.31 | 1.30 |

**The memory's readout |R(m)| (normalised units) under full observation**

| band \ lon | 30E | 90E | 150E | 210E | 270E | 330E |
|---|---|---|---|---|---|---|
| +49 | 0.101 | 0.107 | 0.117 | 0.111 | 0.123 | 0.113 |
| +14 | 0.152 | 0.153 | 0.157 | 0.146 | 0.153 | 0.147 |
| -14 | 0.152 | 0.150 | 0.146 | 0.144 | 0.141 | 0.150 |
| -49 | 0.168 | 0.165 | 0.157 | 0.151 | 0.143 | 0.155 |


## The game on the training years (2018-19)

15 initialisations (2018-01-01 .. 2019-12-31), 120 h episodes: observations chosen after the first 8 forecasts (48 h), then free. GraphCast-weighted all-point MSE (normalised units) of the forecast before each re-observation; 'free' = mean over the free leads (54-120 h); gains (+ = better) against memory:random (paired, bootstrap 95% over initialisations). Blow-ups are counted and scored 10 from then on.

| run | budget | 6 h | 24 h | 48 h | 72 h | 120 h | free | gain vs random (+ = better) | blow-ups | selection entropy (nat) | rho(freq, log tau) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| memory:policy | 0.75 | 0.112 | 0.277 | 0.265 | 0.270 | 0.669 | 0.384 | +87.0% [+29.7%, +155.5%] | 0 | 3.16 | - |
| memory:random | 0.75 | 0.112 | 0.382 | 1.133 | 1.972 | 4.590 | 2.940 | reference | 6 | 3.18 | - |

Headline RMSE (physical units) at 24 / 72 / 120 h:

| run | z500 | t850 | t2m | u10 |
|---|---|---|---|---|
| memory:policy | 310 / 474 / 1.29e+03 | 2.69 / 3.02 / 5.83 | 3.83 / 4.67 / 9.16 | 2.7 / 2.89 / 4.01 |
| memory:random | 340 / 623 / 1.6e+03 | 2.85 / 3.94 / 6.81 | 3.85 / 5.54 / 10.1 | 3.01 / 3.38 / 4.31 |

## Ablation: the policy without the memory's maps (P-nomem)

31 initialisations (2020-01-01 .. 2020-12-31), 120 h episodes: observations chosen after the first 8 forecasts (48 h), then free. GraphCast-weighted all-point MSE (normalised units) of the forecast before each re-observation; 'free' = mean over the free leads (54-120 h); gains (+ = better) against memory:random (paired, bootstrap 95% over initialisations). Blow-ups are counted and scored 10 from then on.

| run | budget | 6 h | 24 h | 48 h | 72 h | 120 h | free | gain vs random (+ = better) | blow-ups | selection entropy (nat) | rho(freq, log tau) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| memory:policy | 0.75 | 0.113 | 0.283 | 0.273 | 0.278 | 1.453 | 0.513 | +82.4% [+46.5%, +122.0%] | 2 | 3.16 | - |
| memory:random | 0.75 | 0.113 | 0.393 | 1.085 | 1.394 | 5.250 | 2.924 | reference | 15 | 3.18 | - |
| n1:policy | 0.75 | 0.100 | 0.112 | 0.127 | 0.183 | 0.361 | 0.239 | +91.8% [+57.4%, +129.8%] | 0 | 3.11 | - |
| n1:random | 0.75 | 0.100 | 0.107 | 0.107 | 0.160 | 0.338 | 0.216 | +92.6% [+58.2%, +130.6%] | 0 | 3.18 | - |

Headline RMSE (physical units) at 24 / 72 / 120 h:

| run | z500 | t850 | t2m | u10 |
|---|---|---|---|---|
| memory:policy | 317 / 462 / 1.27e+03 | 2.62 / 3.05 / 5.74 | 3.74 / 4.58 / 8.72 | 2.68 / 2.93 / 4.05 |
| memory:random | 349 / 762 / 1.59e+03 | 2.85 / 4.63 / 6.67 | 3.92 / 6.4 / 9.26 | 3.03 / 3.42 / 4.31 |
| n1:policy | 133 / 339 / 755 | 1.14 / 2.23 / 3.94 | 1.89 / 3.19 / 5.06 | 1.74 / 2.47 / 3.59 |
| n1:random | 116 / 270 / 702 | 1.05 / 1.95 / 3.74 | 1.94 / 2.65 / 4.63 | 1.63 / 2.23 / 3.42 |

## The climate frame (2020)

8 initialisations; 10 days of budgeted observations (budget 0.75) under each rule, then 4 weeks free; weekly means scored against ERA5 (GraphCast-weighted all-point MSE) and the climatology; gains (+ = better) against memory:random (paired bootstrap). A run that blows up is scored with the climatology from then on.

| run | week 1 | week 2 | week 3 | week 4 | gain vs random (+ = better), weeks 1, 2, 3, 4 | blow-ups |
|---|---|---|---|---|---|---|
| climatology | 0.095 | 0.100 | 0.102 | 0.098 | -2.7% +0.0% -0.0% -0.0% | 0 |
| memory:policy | 0.092 | 0.100 | 0.102 | 0.098 | +0.0% -0.0% -0.0% -0.0% | 8 |
| memory:random | 0.092 | 0.100 | 0.102 | 0.098 | reference | 8 |
| memory:full | 0.181 | 0.112 | 0.102 | 0.098 | -96.2% -11.1% -0.0% -0.0% | 8 |
| memory:none | 0.095 | 0.100 | 0.102 | 0.098 | -2.7% -0.0% -0.0% -0.0% | 8 |
| n1:policy | 0.140 | 0.208 | 0.102 | 0.098 | -52.3% -107.3% -0.0% -0.0% | 8 |
| n1:random | 0.134 | 0.130 | 0.102 | 0.098 | -45.8% -29.5% -0.0% -0.0% | 8 |

Weekly RMSE (physical units) and ACC, z500 and t2m:

| run | z500 RMSE w1..w4 | t850 RMSE w1..w4 | t2m RMSE w1..w4 | z500 ACC |
|---|---|---|---|---|
| climatology | 581 / 591 / 602 / 578 | 2.13 / 2.15 / 2.29 / 2.22 | 1.76 / 1.82 / 1.88 / 1.81 | 0.00 / 0.00 / 0.00 / 0.00 |
| memory:policy | 428 / 591 / 602 / 578 | 1.9 / 2.15 / 2.29 / 2.22 | 2.34 / 1.82 / 1.88 / 1.81 | 0.70 / -0.00 / -0.00 / -0.01 |
| memory:random | 560 / 591 / 602 / 578 | 2.07 / 2.15 / 2.29 / 2.22 | 1.75 / 1.82 / 1.88 / 1.81 | 0.09 / -0.00 / -0.00 / -0.01 |
| memory:full | 946 / 655 / 602 / 578 | 3.56 / 2.26 / 2.29 / 2.22 | 5.85 / 2.16 / 1.88 / 1.81 | 0.47 / -0.02 / -0.00 / -0.01 |
| memory:none | 581 / 591 / 602 / 578 | 2.13 / 2.15 / 2.29 / 2.22 | 1.76 / 1.82 / 1.88 / 1.81 | -0.00 / -0.00 / -0.00 / -0.01 |
| n1:policy | 730 / 1.09e+03 / 602 / 578 | 3.18 / 3.52 / 2.29 / 2.22 | 4.08 / 4.03 / 1.88 / 1.81 | 0.52 / -0.02 / -0.00 / -0.01 |
| n1:random | 704 / 764 / 602 / 578 | 3.13 / 2.63 / 2.29 / 2.22 | 3.85 / 2.66 / 1.88 / 1.81 | 0.56 / -0.02 / -0.00 / -0.01 |

## Decision rules

- **A1** the policy's free-lead error is >= 5% below the random rule (paired CI excludes 0)  
  → supported (+76.7% [+40.9%, +116.8%])
- **A2** the policy is >= 3% below the best fixed rule  
  → not supported (best fixed rule age: policy gain -37.7%; age 0.495, readout 9.375, tendency 0.538)
- **A3** Spearman(selection frequency, patch log tau) >= +0.3: the policy looks where the memory lasts  
  → not supported: the policy looks where the memory is short (rho -0.59)
- **A4** the memory model is >= 5% below N1 + nudging under the same policy (memory maps zero on N1)  
  → not supported (memory 0.681 vs N1 0.238, gain -186.0%)
- **A5** climate frame: the policy's weeks 1-2 error is >= 3% below random  
  → not supported (weeks 1-2 gain +0.0%; weeks 3-4 gain -0.0%, -0.0%)
- **A6** invariance: rank correlation of the policy's patch frequencies, 2018-19 vs 2020, >= 0.7  
  → supported (r +0.95)
- ablation P-nomem (the policy without the memory's maps) vs random: +82.4% [+46.5%, +122.0%]; with the maps: +76.7% [+40.9%, +116.8%]

## Direction

- The budget's room: full observation 0.239, none 1.440, random at the budget 2.924; the random rule already closes -124% of the gap, so the choice can only move the rest.
- The choice carries information for this model: next, widen the action (which variables to observe, and the memory's timescale as a second action), and train the policy jointly with the memory (the DreamerV3 loop of stage 10) so the memory learns to keep what the policy will not look at again.
- N1 + nudging is at least as good under the same observations: the fine-tuned column (§10.1, the causes of stage 10) still costs more than the memory gains; retrain the memory with the stage 10.2 continuous filter before the next observation-policy round.
- Climate frame: weeks 3-4 gain vs random -0.0%, -0.0%; beyond two weeks the placement no longer matters, as expected without slow memory channels (10.2).
