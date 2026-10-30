## Self-explainability (6 h pairs)

| arm | Lie codes | physics codes | loss: persistence | MDL-selected | full K | floor | explained fraction | mean k* | codes used / perplexity | physics share |
|---|---|---|---|---|---|---|---|---|---|---|
| C (wb2_64x32_theory_vq_lie) | 16 | 0 | 0.7642 | 0.6574 | 0.6623 | 0.5302 | 0.452 | 2.16 | 16 / 5.7 | – |
| A' (wb2_64x32_theory_continuous) | cont. | 0 | 0.7642 | 0.6497 | 0.6493 | 0.5302 | 0.485 | 2.69 | continuous | – |
| E1 (wb2_64x32_theory_hybrid) | 16 | 15 | 0.7642 | 0.6835 | 0.7466 | 0.5302 | 0.340 | 0.85 | 16 / 8.7 | 0% |
| E2 (wb2_64x32_theory_hybrid) | 16 | 15 | 0.7642 | 0.6584 | 0.6581 | 0.5302 | 0.447 | 1.51 | 16 / 4.8 | -46% |
| D (wb2_64x32_theory_physics_only) | 1 | 15 | 0.7642 | 0.7305 | 0.7019 | 0.5302 | 0.141 | 0.87 | 1 / 1.0 | 100% |

## Analog transfer (program of pair i executed on its nearest state j ≥ 7 days away)

| arm | persistence | delta copy | random program | **transferred** | oracle | transfer gain | delta gain | transfer beats delta |
|---|---|---|---|---|---|---|---|---|
| C | 0.7349 | 1.0285 | 0.6606 | **0.6534** | 0.6380 | 0.840 | -3.198 | 100% |
| A' | 0.7349 | 1.0285 | 0.6520 | **0.6446** | 0.6303 | 0.861 | -2.959 | 100% |
| E1 | 0.7349 | 1.0285 | 0.6973 | **0.6902** | 0.6762 | 0.765 | -4.457 | 100% |
| E2 | 0.7349 | 1.0285 | 0.6591 | **0.6547** | 0.6415 | 0.862 | -3.319 | 100% |
| D | 0.7349 | 1.0285 | 0.7193 | **0.7170** | 0.7144 | 0.982 | -5.000 | 100% |

## Length generalisation (12 h pairs, trained on 6 h)

| arm | max steps | persistence | selected | full | floor | mean k* | explained fraction |
|---|---|---|---|---|---|---|---|
| C | 3 | 1.3984 | 1.0044 | 1.0047 | 0.5533 | 2.73 | 0.463 |
| C | 6 | 1.3984 | 0.9760 | 0.9862 | 0.5533 | 4.26 | 0.496 |
| A' | 3 | 1.3984 | 1.0131 | 1.0128 | 0.5533 | 2.95 | 0.452 |
| A' | 6 | 1.3984 | 0.9534 | 0.9539 | 0.5533 | 5.15 | 0.523 |
| E1 | 3 | 1.3984 | 1.0517 | 1.0786 | 0.5533 | 1.51 | 0.406 |
| E1 | 6 | 1.3984 | 1.0437 | 1.2228 | 0.5533 | 1.73 | 0.416 |
| E2 | 3 | 1.3984 | 1.0083 | 1.0031 | 0.5533 | 2.25 | 0.458 |
| E2 | 6 | 1.3984 | 0.9838 | 0.9936 | 0.5533 | 2.99 | 0.487 |
| D | 3 | 1.3984 | 1.3191 | 1.2871 | 0.5533 | 1.36 | 0.092 |
| D | 6 | 1.3984 | 1.2992 | 1.2523 | 0.5533 | 1.99 | 0.116 |

## Physics codes: usage and intervention tests

| arm | physics share of steps / of error reduction | usage per shell (surface → top) | usage tropics / extratropics | Lie usage tropics / extratropics |
|---|---|---|---|---|
| E1 | 100% / 0% | 100%, 100%, 100%, 100%, 100% | 100% / 100% | 72% / 79% |
| E2 | 19% / -46% | 1%, 95%, 0%, 0%, 0% | 20% / 19% | 52% / 41% |
| D | 34% / 100% | 93%, 0%, 62%, 15%, 1% | 29% / 38% | 0% / 0% |

| arm | TISR := night: latent effect where radiation / mixing / surface / no code was used | decoded 2 m temperature change | knock-out loss increase: radiation / mixing / surface |
|---|---|---|---|
| E1 | 3.3e-04 / 3.7e-05 / 4.3e-06 / 1.3e-05 | -0.0030 | +0.22% / +0.31% / +0.00% |
| E2 | 3.7e-03 / 1.3e-04 / 1.5e-04 / 9.0e-05 | -0.0021 | +0.16% / +0.47% / +0.09% |
| D | 3.3e-05 / 7.2e-07 / 3.1e-05 / 1.0e-05 | -0.0001 | +0.29% / +2.84% / +0.01% |

## Cross-year invariance (2015–17 vocabulary applied to 2020 pairs)

| arm | TV distance of Lie code usage (2018 vs 2020) | TV of physics usage per shell | mean k* 2018 → 2020 | explained fraction 2018 → 2020 | transfer gain 2018 → 2020 |
|---|---|---|---|---|---|
| C | 0.012 | – | 2.16 → 2.13 | 0.452 → 0.457 | 0.840 → 0.845 |
| E2 | 0.011 | 0.000 | 1.51 → 1.46 | 0.447 → 0.452 | 0.862 → 0.861 |
