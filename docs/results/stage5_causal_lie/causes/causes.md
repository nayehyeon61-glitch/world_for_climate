# Cause analysis: where the forecast error comes from

Arms S, P, P-skip, P-nodrop, G1; initialisations {'S': 31, 'P': 31, 'P-skip': 31, 'P-nodrop': 31, 'G1': 31}; leads [6, 24, 72] h.

## 1. Error decomposition: total = autoencoder floor (rec) + latent dynamics (dyn)

RMSE in physical units (scaled as in the stage 4 report). `rec share` = rec² / total², the part of the squared error that no transition could remove with this autoencoder.

| variable | lead | S total / rec / dyn (rec share) | P total / rec / dyn (rec share) | P-skip total / rec / dyn (rec share) | P-nodrop total / rec / dyn (rec share) | G1 total / rec / dyn (rec share) |
|---|---|---|---|---|---|---|
| geopotential@500 (m²/s²) | 6h | 264.2 / 253.6 / 193 (0.92) | 357.1 / 360.8 / 220 (1.02) | 430.8 / 451 / 181.7 (1.09) | 280.3 / 270.7 / 190 (0.93) | 356.8 / 355.9 / 219.8 (0.99) |
| geopotential@500 (m²/s²) | 24h | 444 / 254 / 450 (0.33) | 485 / 363.5 / 478.4 (0.56) | 554.1 / 458.8 / 515.4 (0.68) | 449.5 / 277.8 / 435.6 (0.38) | 499.1 / 361.3 / 465.9 (0.52) |
| geopotential@500 (m²/s²) | 72h | 816.5 / 259.5 / 813.5 (0.10) | 862 / 371 / 864.5 (0.18) | 813.8 / 460.4 / 748.5 (0.32) | 888 / 279.5 / 872.1 (0.10) | 867.7 / 367.4 / 840.3 (0.18) |
| temperature@850 (K) | 6h | 1.93 / 1.909 / 0.9824 (0.98) | 2.405 / 2.429 / 0.9959 (1.02) | 2.838 / 2.883 / 0.7014 (1.03) | 2.033 / 2.023 / 0.9547 (0.99) | 2.522 / 2.523 / 0.9254 (1.00) |
| temperature@850 (K) | 24h | 2.568 / 1.947 / 1.918 (0.58) | 2.708 / 2.469 / 1.595 (0.83) | 3.06 / 2.908 / 1.763 (0.90) | 2.465 / 2.05 / 1.777 (0.69) | 2.784 / 2.547 / 1.64 (0.84) |
| temperature@850 (K) | 72h | 3.689 / 1.96 / 3.185 (0.28) | 3.674 / 2.479 / 2.79 (0.45) | 3.79 / 2.937 / 2.638 (0.60) | 3.636 / 2.048 / 3.109 (0.32) | 3.678 / 2.557 / 2.886 (0.48) |
| specific_humidity@700 (g/kg) | 6h | 0.619 / 0.5875 / 0.4029 (0.90) | 0.8421 / 0.8379 / 0.3631 (0.99) | 1.142 / 1.165 / 0.244 (1.04) | 0.6562 / 0.6286 / 0.4189 (0.92) | 0.8505 / 0.8449 / 0.3889 (0.99) |
| specific_humidity@700 (g/kg) | 24h | 0.9609 / 0.5888 / 0.8855 (0.38) | 1.026 / 0.8437 / 0.7468 (0.68) | 1.208 / 1.171 / 0.6266 (0.94) | 0.9612 / 0.6332 / 0.8502 (0.43) | 1.032 / 0.8506 / 0.7639 (0.68) |
| specific_humidity@700 (g/kg) | 72h | 1.43 / 0.5909 / 1.386 (0.17) | 1.421 / 0.8458 / 1.209 (0.35) | 1.433 / 1.166 / 0.9804 (0.66) | 1.513 / 0.6342 / 1.429 (0.18) | 1.479 / 0.8511 / 1.285 (0.33) |
| u_component_of_wind@850 (m/s) | 6h | 2.16 / 1.86 / 1.432 (0.74) | 2.282 / 2.007 / 1.524 (0.77) | 3.2 / 3.224 / 1.023 (1.02) | 2.185 / 1.865 / 1.511 (0.73) | 2.241 / 2.011 / 1.431 (0.81) |
| u_component_of_wind@850 (m/s) | 24h | 3.707 / 1.858 / 3.423 (0.25) | 3.702 / 2.002 / 3.463 (0.29) | 4.026 / 3.235 / 2.829 (0.64) | 3.751 / 1.856 / 3.514 (0.24) | 3.639 / 2.008 / 3.341 (0.30) |
| u_component_of_wind@850 (m/s) | 72h | 5.891 / 1.867 / 5.633 (0.10) | 6.004 / 2.02 / 5.841 (0.11) | 5.562 / 3.262 / 4.505 (0.34) | 6.064 / 1.865 / 5.844 (0.09) | 5.62 / 2.024 / 5.374 (0.13) |
| 2m_temperature (K) | 6h | 2.166 / 2.419 / 1.316 (1.24) | 2.442 / 2.669 / 1.286 (1.19) | 3.295 / 3.377 / 0.7767 (1.05) | 2.163 / 2.427 / 1.274 (1.25) | 2.552 / 2.655 / 1.199 (1.08) |
| 2m_temperature (K) | 24h | 3.101 / 2.533 / 2.06 (0.67) | 3.213 / 2.84 / 1.713 (0.78) | 3.604 / 3.544 / 1.658 (0.97) | 3.011 / 2.564 / 1.987 (0.73) | 3.135 / 2.768 / 1.81 (0.78) |
| 2m_temperature (K) | 72h | 4.431 / 2.555 / 3.934 (0.33) | 4.216 / 2.841 / 3.195 (0.45) | 4.043 / 3.543 / 2.537 (0.77) | 4.567 / 2.582 / 4.131 (0.32) | 4.381 / 2.775 / 3.613 (0.40) |
| mean_sea_level_pressure (hPa) | 6h | 2.36 / 2.049 / 1.909 (0.75) | 2.96 / 2.722 / 2.011 (0.84) | 4.034 / 4.006 / 1.683 (0.98) | 2.357 / 1.91 / 1.895 (0.66) | 2.919 / 2.614 / 1.976 (0.80) |
| mean_sea_level_pressure (hPa) | 24h | 4.544 / 2.092 / 4.541 (0.21) | 4.707 / 2.716 / 4.413 (0.33) | 5.495 / 4.027 / 4.448 (0.54) | 4.469 / 1.94 / 4.466 (0.19) | 4.754 / 2.665 / 4.49 (0.31) |
| mean_sea_level_pressure (hPa) | 72h | 7.421 / 2.084 / 7.313 (0.08) | 7.862 / 2.724 / 7.605 (0.12) | 7.895 / 4.019 / 7.108 (0.26) | 7.788 / 1.953 / 7.795 (0.06) | 8.371 / 2.687 / 8.296 (0.10) |
| 10m_u_component_of_wind (m/s) | 6h | 1.572 / 1.317 / 1.028 (0.70) | 1.643 / 1.401 / 1.119 (0.73) | 2.315 / 2.315 / 0.7786 (1.00) | 1.583 / 1.314 / 1.068 (0.69) | 1.631 / 1.398 / 1.04 (0.73) |
| 10m_u_component_of_wind (m/s) | 24h | 2.619 / 1.31 / 2.441 (0.25) | 2.635 / 1.389 / 2.483 (0.28) | 2.898 / 2.312 / 2.045 (0.63) | 2.632 / 1.302 / 2.482 (0.24) | 2.619 / 1.385 / 2.417 (0.28) |
| 10m_u_component_of_wind (m/s) | 72h | 4.034 / 1.323 / 3.882 (0.11) | 4.046 / 1.408 / 3.978 (0.12) | 3.903 / 2.34 / 3.194 (0.36) | 4.111 / 1.312 / 3.985 (0.10) | 3.966 / 1.405 / 3.845 (0.13) |
| total_precipitation_6hr (mm) | 6h | 0.6652 / 0.58 / 0.5601 (0.76) | 0.7209 / 0.6715 / 0.5369 (0.87) | 1.088 / 1.1 / 0.2191 (1.02) | 0.6728 / 0.5662 / 0.5472 (0.71) | 0.7406 / 0.6418 / 0.5327 (0.75) |
| total_precipitation_6hr (mm) | 24h | 1.163 / 0.5871 / 1.052 (0.26) | 1.171 / 0.6895 / 1 (0.35) | 1.215 / 1.127 / 0.5278 (0.86) | 1.157 / 0.578 / 1.047 (0.25) | 1.161 / 0.6622 / 0.9864 (0.33) |
| total_precipitation_6hr (mm) | 72h | 1.411 / 0.5893 / 1.293 (0.18) | 1.408 / 0.6996 / 1.244 (0.25) | 1.361 / 1.136 / 0.7608 (0.70) | 1.413 / 0.5802 / 1.312 (0.17) | 1.39 / 0.6706 / 1.221 (0.23) |

Mean rec share over all variables:

| arm | 6h | 24h | 72h |
|---|---|---|---|
| S | 0.83 | 0.40 | 0.20 |
| P | 0.90 | 0.54 | 0.29 |
| P-skip | 1.04 | 0.80 | 0.55 |
| P-nodrop | 0.82 | 0.42 | 0.20 |
| G1 | 0.90 | 0.55 | 0.31 |

## 2. Eval-time ablations (same states and noise, paired over initialisations)

RMSE change of the ablated rollout against the full model, mean and 90% bootstrap interval. Positive = the part helps (removing it hurts).

| arm | ablation | variable | 6h | 24h | 72h |
|---|---|---|---|---|---|
| S | no denoiser (group action only) | geopotential@500 | **-3.6%** [-4.0, -3.1] | **-3.4%** [-4.0, -2.8] | **-2.7%** [-3.6, -1.9] |
| S | no denoiser (group action only) | temperature@850 | **-2.2%** [-2.6, -1.9] | **-3.3%** [-3.8, -2.7] | **-8.0%** [-9.1, -7.0] |
| S | no denoiser (group action only) | specific_humidity@700 | **-4.7%** [-5.0, -4.4] | **-5.2%** [-5.7, -4.7] | **-5.0%** [-5.5, -4.5] |
| S | no denoiser (group action only) | u_component_of_wind@850 | **-5.0%** [-5.2, -4.7] | **-6.9%** [-7.4, -6.3] | **-10.7%** [-11.6, -9.9] |
| S | no denoiser (group action only) | 2m_temperature | **-8.2%** [-8.6, -7.8] | **-4.7%** [-5.6, -3.9] | **-9.5%** [-11.0, -8.1] |
| S | no denoiser (group action only) | mean_sea_level_pressure | **-6.1%** [-6.6, -5.7] | **-4.6%** [-5.2, -4.0] | **-5.2%** [-6.1, -4.4] |
| S | no denoiser (group action only) | 10m_u_component_of_wind | **-3.7%** [-4.0, -3.5] | **-5.2%** [-5.8, -4.6] | **-9.6%** [-10.3, -9.0] |
| S | no denoiser (group action only) | total_precipitation_6hr | **-5.1%** [-5.5, -4.6] | **-2.8%** [-3.3, -2.4] | **-4.1%** [-4.5, -3.6] |
| P | no denoiser (group action only) | geopotential@500 | **-3.0%** [-3.4, -2.7] | **-4.6%** [-5.2, -3.8] | **-6.3%** [-7.5, -5.0] |
| P | no denoiser (group action only) | temperature@850 | **-1.1%** [-1.3, -0.8] | **-0.8%** [-1.3, -0.3] | **-3.9%** [-5.2, -2.7] |
| P | no denoiser (group action only) | specific_humidity@700 | **-2.5%** [-2.7, -2.2] | **-4.3%** [-4.8, -3.9] | **-5.9%** [-6.5, -5.3] |
| P | no denoiser (group action only) | u_component_of_wind@850 | **-4.0%** [-4.3, -3.8] | **-5.9%** [-6.4, -5.3] | **-11.3%** [-12.3, -10.3] |
| P | no denoiser (group action only) | 2m_temperature | **-4.2%** [-4.5, -4.0] | **-5.0%** [-6.0, -4.1] | **-12.2%** [-14.1, -10.4] |
| P | no denoiser (group action only) | mean_sea_level_pressure | **-3.9%** [-4.3, -3.5] | **-3.9%** [-4.7, -3.1] | **-7.5%** [-8.9, -6.1] |
| P | no denoiser (group action only) | 10m_u_component_of_wind | **-3.5%** [-3.8, -3.3] | **-4.6%** [-5.1, -4.1] | **-9.6%** [-10.4, -8.9] |
| P | no denoiser (group action only) | total_precipitation_6hr | **-3.2%** [-3.5, -2.9] | **-2.6%** [-3.1, -2.1] | **-4.1%** [-4.9, -3.4] |
| P-skip | no denoiser (group action only) | geopotential@500 | **-0.8%** [-1.1, -0.5] | **+2.0%** [+1.4, +2.7] | **+1.5%** [+0.5, +2.5] |
| P-skip | no denoiser (group action only) | temperature@850 | **-0.5%** [-0.8, -0.3] | -0.0% [-0.7, +0.7] | +0.1% [-1.1, +1.2] |
| P-skip | no denoiser (group action only) | specific_humidity@700 | **-0.7%** [-0.8, -0.5] | **-1.6%** [-2.0, -1.2] | **-2.0%** [-2.7, -1.3] |
| P-skip | no denoiser (group action only) | u_component_of_wind@850 | **-0.9%** [-1.1, -0.7] | **-0.6%** [-1.1, -0.2] | -0.6% [-1.2, +0.1] |
| P-skip | no denoiser (group action only) | 2m_temperature | **-0.9%** [-1.1, -0.6] | **-1.1%** [-1.6, -0.5] | **-1.9%** [-3.0, -0.7] |
| P-skip | no denoiser (group action only) | mean_sea_level_pressure | **-0.9%** [-1.3, -0.6] | -0.5% [-1.0, +0.1] | **-2.0%** [-2.8, -1.2] |
| P-skip | no denoiser (group action only) | 10m_u_component_of_wind | **-0.9%** [-1.1, -0.8] | **-1.0%** [-1.5, -0.5] | **-1.4%** [-2.1, -0.8] |
| P-skip | no denoiser (group action only) | total_precipitation_6hr | **-0.3%** [-0.5, -0.2] | **-0.5%** [-0.8, -0.2] | **-0.9%** [-1.3, -0.5] |
| P-nodrop | no denoiser (group action only) | geopotential@500 | **-3.9%** [-4.4, -3.3] | **-4.2%** [-5.0, -3.3] | +1.1% [-0.1, +2.5] |
| P-nodrop | no denoiser (group action only) | temperature@850 | **-2.2%** [-2.6, -1.9] | **-4.1%** [-4.5, -3.6] | **-6.6%** [-7.7, -5.6] |
| P-nodrop | no denoiser (group action only) | specific_humidity@700 | **-4.5%** [-4.7, -4.3] | **-7.2%** [-7.8, -6.7] | **-10.8%** [-11.6, -9.9] |
| P-nodrop | no denoiser (group action only) | u_component_of_wind@850 | **-6.7%** [-7.0, -6.4] | **-8.4%** [-9.0, -7.9] | **-12.2%** [-13.2, -11.2] |
| P-nodrop | no denoiser (group action only) | 2m_temperature | **-7.3%** [-7.8, -6.9] | **-2.0%** [-2.5, -1.4] | **-8.8%** [-10.5, -7.2] |
| P-nodrop | no denoiser (group action only) | mean_sea_level_pressure | **-5.7%** [-6.1, -5.3] | **-5.1%** [-5.9, -4.3] | **-6.3%** [-7.8, -4.9] |
| P-nodrop | no denoiser (group action only) | 10m_u_component_of_wind | **-5.3%** [-5.6, -5.0] | **-6.4%** [-6.8, -6.0] | **-9.9%** [-10.9, -8.8] |
| P-nodrop | no denoiser (group action only) | total_precipitation_6hr | **-5.0%** [-5.5, -4.6] | **-2.6%** [-3.2, -2.1] | **-3.9%** [-4.6, -3.1] |
| G1 | no denoiser (group action only) | geopotential@500 | **-3.0%** [-3.4, -2.6] | **-2.2%** [-3.1, -1.4] | **-3.9%** [-5.0, -2.8] |
| G1 | no denoiser (group action only) | temperature@850 | **-1.7%** [-2.0, -1.5] | **-1.5%** [-2.0, -1.0] | **+1.3%** [+0.3, +2.3] |
| G1 | no denoiser (group action only) | specific_humidity@700 | **-2.3%** [-2.5, -2.1] | **-4.0%** [-4.4, -3.7] | **-8.3%** [-9.0, -7.6] |
| G1 | no denoiser (group action only) | u_component_of_wind@850 | **-2.7%** [-3.0, -2.5] | **-2.2%** [-2.6, -1.8] | **-4.8%** [-5.4, -4.3] |
| G1 | no denoiser (group action only) | 2m_temperature | **-3.1%** [-3.4, -2.8] | **-2.0%** [-3.0, -1.0] | **-3.8%** [-5.9, -1.6] |
| G1 | no denoiser (group action only) | mean_sea_level_pressure | **-4.9%** [-5.3, -4.5] | **-4.3%** [-5.4, -3.2] | **-5.9%** [-8.1, -3.9] |
| G1 | no denoiser (group action only) | 10m_u_component_of_wind | **-2.9%** [-3.2, -2.7] | **-3.3%** [-3.8, -2.8] | **-7.2%** [-7.8, -6.5] |
| G1 | no denoiser (group action only) | total_precipitation_6hr | **-3.0%** [-3.4, -2.7] | **-2.9%** [-3.5, -2.3] | **-3.9%** [-4.4, -3.4] |
| G1 | composite group off | geopotential@500 | **+0.3%** [+0.1, +0.5] | +0.0% [-0.6, +0.7] | **-2.2%** [-3.8, -0.7] |
| G1 | composite group off | temperature@850 | **+0.8%** [+0.6, +0.9] | **+5.7%** [+5.1, +6.3] | **+21.2%** [+19.0, +23.4] |
| G1 | composite group off | specific_humidity@700 | **-0.2%** [-0.3, -0.1] | +0.2% [-0.1, +0.5] | **+2.8%** [+2.2, +3.5] |
| G1 | composite group off | u_component_of_wind@850 | +0.0% [-0.0, +0.1] | +0.1% [-0.0, +0.2] | -0.1% [-0.3, +0.2] |
| G1 | composite group off | 2m_temperature | **+1.2%** [+1.0, +1.3] | **+1.8%** [+1.3, +2.4] | **+7.8%** [+6.3, +9.4] |
| G1 | composite group off | mean_sea_level_pressure | **+1.1%** [+0.8, +1.4] | **+4.3%** [+3.5, +5.1] | **+10.5%** [+8.8, +12.4] |
| G1 | composite group off | 10m_u_component_of_wind | -0.0% [-0.0, +0.0] | +0.1% [-0.0, +0.2] | **+0.3%** [+0.0, +0.5] |
| G1 | composite group off | total_precipitation_6hr | **+0.8%** [+0.6, +1.0] | **+3.0%** [+2.7, +3.3] | **+7.7%** [+6.7, +8.7] |
| G1 | vertical shift off | geopotential@500 | +0.0% [-0.0, +0.0] | **+0.2%** [+0.1, +0.2] | **+1.3%** [+1.1, +1.5] |
| G1 | vertical shift off | temperature@850 | **+0.0%** [+0.0, +0.0] | +0.0% [-0.0, +0.1] | +0.1% [-0.1, +0.2] |
| G1 | vertical shift off | specific_humidity@700 | **-0.1%** [-0.1, -0.1] | **-0.2%** [-0.3, -0.2] | **+0.3%** [+0.2, +0.4] |
| G1 | vertical shift off | u_component_of_wind@850 | -0.0% [-0.0, +0.0] | +0.1% [-0.0, +0.1] | **+0.7%** [+0.6, +0.8] |
| G1 | vertical shift off | 2m_temperature | **+0.1%** [+0.1, +0.1] | +0.0% [-0.1, +0.1] | **+0.6%** [+0.3, +0.8] |
| G1 | vertical shift off | mean_sea_level_pressure | **+0.1%** [+0.0, +0.1] | **+0.2%** [+0.1, +0.3] | **+1.3%** [+1.1, +1.5] |
| G1 | vertical shift off | 10m_u_component_of_wind | **-0.1%** [-0.1, -0.0] | **+0.2%** [+0.1, +0.3] | **+1.1%** [+1.0, +1.3] |
| G1 | vertical shift off | total_precipitation_6hr | +0.0% [-0.0, +0.0] | **+0.1%** [+0.0, +0.1] | +0.0% [-0.0, +0.1] |
| G1 | mixing off | geopotential@500 | +0.0% [+0.0, +0.0] | +0.0% [+0.0, +0.0] | +0.0% [+0.0, +0.0] |
| G1 | mixing off | temperature@850 | +0.0% [+0.0, +0.0] | +0.0% [+0.0, +0.0] | +0.0% [+0.0, +0.0] |
| G1 | mixing off | specific_humidity@700 | +0.0% [+0.0, +0.0] | +0.0% [+0.0, +0.0] | +0.0% [+0.0, +0.0] |
| G1 | mixing off | u_component_of_wind@850 | +0.0% [+0.0, +0.0] | +0.0% [+0.0, +0.0] | +0.0% [+0.0, +0.0] |
| G1 | mixing off | 2m_temperature | +0.0% [+0.0, +0.0] | +0.0% [+0.0, +0.0] | +0.0% [+0.0, +0.0] |
| G1 | mixing off | mean_sea_level_pressure | +0.0% [+0.0, +0.0] | +0.0% [+0.0, +0.0] | +0.0% [+0.0, +0.0] |
| G1 | mixing off | 10m_u_component_of_wind | +0.0% [+0.0, +0.0] | +0.0% [+0.0, +0.0] | +0.0% [+0.0, +0.0] |
| G1 | mixing off | total_precipitation_6hr | +0.0% [+0.0, +0.0] | +0.0% [+0.0, +0.0] | +0.0% [+0.0, +0.0] |
