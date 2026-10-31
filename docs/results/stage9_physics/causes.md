# Stage 9: the coupled physics of N1 under interventions

Model: `runs/physics_n1/checkpoint.pt`; 8 initialisations per year (2018, 2020), 72 h, the same dates in N0. Changes: channel-mean normalised MSE vs the intact model (+ = worse without the arrow). Blow-ups: runs that diverged by 72 h (per year).

## The arrows

| intervention | arrow | 2018 6 h | 2018 24 h | 2018 72 h | 2020 6 h | 2020 24 h | 2020 72 h | blow-ups | N0 72 h |
|---|---|---|---|---|---|---|---|---|---|
| `no_temp_advection` | motion -> thermodynamics (u, omega advect T) | +108% | +74% | +26% | +104% | +73% | +31% | 0/0 | -11% / -21% |
| `no_adiabatic` | motion -> thermodynamics (kappa T omega / p) | blown | blown | blown | blown | blown | blown | 8/8 | blown / blown |
| `no_density` | thermodynamics -> motion (T' in the pressure gradient: density) | +6282% | blown | blown | +5981% | blown | blown | 8/8 | blown / blown |
| `weak_density` | thermodynamics -> motion, T' coupling 10% weaker (soft: the full cut blows up) | +12% | +8% | +1% | +11% | +7% | +1% | 0/0 | +2% / +2% |
| `no_virtual` | water -> hydrostatic balance (T_v: buoyancy of water vapour) | +3% | +0% | -2% | +3% | +1% | -2% | 0/0 | -3% / -4% |
| `no_moisture` | motion -> water (u, omega advect q) | +8% | +17% | +1% | +9% | +19% | +2% | 0/0 | -26% / -36% |
| `no_pgf` | hydrostatic balance -> motion (the whole pressure gradient: Phi, Phi_b) | blown | blown | blown | blown | blown | blown | 8/8 | blown / blown |
| `no_advection` | motion -> motion (horizontal advection of momentum, T and Phi_b) | +132% | +149% | +58% | +133% | +146% | +63% | 0/0 | +18% / +10% |
| `no_coriolis` | rotation -> motion (Coriolis) | +2867% | +2995% | +1425% | +2880% | +2807% | +1668% | 7/7 | blown / blown |
| `no_vertical` | motion -> all (vertical advection) | +73% | +47% | +16% | +68% | +45% | +20% | 0/0 | -20% / -31% |
| `no_drag` | surface -> motion (boundary-layer friction) | +2% | +5% | +17% | +2% | +5% | +17% | 0/0 | +77% / +36% |
| `no_carry` | initial residual (unresolved scales, offsets) | +0% | -2% | -1% | +1% | -2% | -1% | 0/0 | -1% / -1% |
| `no_heating` | radiation + latent heating -> thermodynamics (column Q) | +61% | +54% | +29% | +60% | +56% | +36% | 0/0 | n/a / n/a |
| `no_moistening` | water sources and sinks (column S) | +16% | blown | blown | +21% | blown | blown | 8/8 | n/a / n/a |
| `no_momentum` | column momentum forcing -> motion | +11% | +10% | +19% | +11% | +9% | +18% | 0/0 | n/a / n/a |
| `no_surface_pressure` | column -> surface pressure (P_b) | +3% | +0% | -1% | +3% | +1% | +0% | 0/0 | n/a / n/a |
| `q_to_heating` | water -> thermodynamics (latent heating) and water -> radiation (moisture) | +23% | +17% | +3% | +21% | +16% | +5% | 0/0 | n/a / n/a |
| `t_to_moistening` | thermodynamics -> water (saturation, condensation) | +16% | +22% | +12% | +14% | +21% | +16% | 0/0 | n/a / n/a |
| `sun_off` | sun -> radiation: do(TISR = 0) | +0% | +1% | +0% | +0% | +1% | +1% | 0/0 | n/a / n/a |
| `sun_to_heating` | radiation -> thermodynamics (heating and cooling), the sun's path only | +0% | +0% | -0% | +0% | +0% | -0% | 0/0 | n/a / n/a |
| `ocean_column` | surface (land / ocean) -> radiation and column physics | +4% | +4% | -1% | +3% | +3% | -0% | 0/0 | n/a / n/a |

Arrows without a path in N1:

- surface <-> water (precipitation, evaporation): evaporation E only diagnoses precipitation (P = E - int S dp / g); no tendency reads E, and nothing stores surface water or heat

## Per variable at 72 h, 2018

| intervention | z500 | t850 | q700 | u850 | t2m | msl | u10 | tp |
|---|---|---|---|---|---|---|---|---|
| `no_temp_advection` | +37% | +10% | +2% | +17% | -27% | +30% | +16% | +0% |
| `no_adiabatic` | blown | blown | blown | blown | blown | blown | blown | blown |
| `no_density` | blown | blown | blown | blown | blown | blown | blown | blown |
| `weak_density` | +5% | +0% | +0% | -0% | -2% | +4% | +0% | -0% |
| `no_virtual` | +3% | +0% | -2% | +0% | -2% | +2% | -0% | -2% |
| `no_moisture` | +15% | +1% | +6% | +3% | +1% | +6% | +1% | +2% |
| `no_pgf` | blown | blown | blown | blown | blown | blown | blown | blown |
| `no_advection` | +116% | +27% | +14% | +40% | -21% | +87% | +33% | +4% |
| `no_coriolis` | +420% | +202% | +165% | +146% | +53% | +172% | +64% | +16% |
| `no_vertical` | +40% | +9% | +7% | +17% | -9% | +25% | +15% | +3% |
| `no_drag` | +24% | +8% | +5% | +23% | +8% | +26% | +24% | +5% |
| `no_carry` | -0% | -1% | -2% | -1% | +33% | +6% | -4% | +0% |
| `no_heating` | -14% | +1% | +5% | +15% | +44% | +15% | +15% | +2% |
| `no_moistening` | blown | blown | blown | blown | blown | blown | blown | blown |
| `no_momentum` | +57% | +21% | +2% | +17% | +10% | +54% | +24% | +2% |
| `no_surface_pressure` | +3% | -3% | -0% | -1% | +2% | +33% | +1% | -2% |
| `q_to_heating` | +7% | +20% | +1% | +15% | -9% | +34% | +12% | -1% |
| `t_to_moistening` | -14% | -6% | +21% | -0% | -3% | -4% | -0% | -0% |
| `sun_off` | +19% | +13% | -1% | +1% | +14% | +4% | +0% | +1% |
| `sun_to_heating` | +6% | +12% | +0% | +0% | +11% | -2% | +0% | +1% |
| `ocean_column` | -15% | +3% | -0% | +1% | +2% | -2% | +5% | -1% |
| `no_temp_advection+weak_density` | +39% | +11% | +2% | +18% | -30% | +29% | +17% | +0% |
| `no_moisture+no_virtual` | +16% | +1% | +6% | +3% | -0% | +7% | +1% | +2% |
| `no_moisture+q_to_heating` | +10% | +20% | +3% | +14% | -9% | +38% | +12% | +2% |
| `t_to_moistening+q_to_heating` | +3% | +20% | +27% | +14% | -8% | +21% | +13% | +1% |
| `sun_off+no_heating` | -10% | +1% | +4% | +16% | +46% | +20% | +15% | +2% |
| `sun_off+q_to_heating` | +32% | +39% | +1% | +16% | +2% | +40% | +14% | -0% |
| `sun_off+ocean_column` | -13% | +11% | -2% | +1% | +13% | -5% | +4% | -1% |

## Per variable at 72 h, 2020

| intervention | z500 | t850 | q700 | u850 | t2m | msl | u10 | tp |
|---|---|---|---|---|---|---|---|---|
| `no_temp_advection` | +38% | +16% | +2% | +17% | -24% | +34% | +18% | -0% |
| `no_adiabatic` | blown | blown | blown | blown | blown | blown | blown | blown |
| `no_density` | blown | blown | blown | blown | blown | blown | blown | blown |
| `weak_density` | +5% | +1% | +0% | -0% | -2% | +4% | +1% | -0% |
| `no_virtual` | +3% | +1% | -2% | -0% | -2% | +2% | +0% | -2% |
| `no_moisture` | +14% | +2% | +3% | +2% | +1% | +6% | +1% | -0% |
| `no_pgf` | blown | blown | blown | blown | blown | blown | blown | blown |
| `no_advection` | +111% | +31% | +13% | +37% | -18% | +91% | +33% | +4% |
| `no_coriolis` | +492% | +186% | +189% | +185% | +52% | +405% | +89% | +31% |
| `no_vertical` | +42% | +12% | +4% | +16% | -10% | +28% | +15% | +1% |
| `no_drag` | +24% | +8% | +5% | +26% | +8% | +25% | +27% | +4% |
| `no_carry` | -0% | -1% | -2% | -1% | +38% | +4% | -3% | +0% |
| `no_heating` | -12% | -0% | +3% | +13% | +46% | +16% | +14% | -1% |
| `no_moistening` | blown | blown | blown | blown | blown | blown | blown | blown |
| `no_momentum` | +58% | +22% | +1% | +12% | +11% | +53% | +23% | +2% |
| `no_surface_pressure` | +8% | -5% | -0% | -1% | +1% | +42% | +1% | -2% |
| `q_to_heating` | +10% | +20% | -1% | +12% | -8% | +32% | +14% | -3% |
| `t_to_moistening` | -10% | -8% | +19% | -1% | -2% | -2% | -0% | -1% |
| `sun_off` | +17% | +13% | -0% | +1% | +13% | +3% | +0% | +1% |
| `sun_to_heating` | +5% | +13% | +1% | +0% | +10% | -2% | +0% | +1% |
| `ocean_column` | -10% | +0% | -0% | +0% | -1% | +2% | +6% | -0% |
| `no_temp_advection+weak_density` | +40% | +16% | +2% | +18% | -26% | +33% | +19% | -1% |
| `no_moisture+no_virtual` | +16% | +2% | +3% | +2% | -0% | +8% | +2% | -0% |
| `no_moisture+q_to_heating` | +14% | +20% | +1% | +12% | -8% | +35% | +13% | -0% |
| `t_to_moistening+q_to_heating` | +8% | +20% | +23% | +12% | -7% | +21% | +14% | -1% |
| `sun_off+no_heating` | -9% | -0% | +1% | +13% | +48% | +20% | +14% | -0% |
| `sun_off+q_to_heating` | +34% | +40% | -1% | +14% | +3% | +37% | +15% | -2% |
| `sun_off+ocean_column` | -9% | +9% | -2% | +1% | +9% | -1% | +5% | +0% |

## Propagation (2018): RMS response vs the intact model, normalised units

| intervention | thermodynamics 6/24/72 h | momentum 6/24/72 h | hydrostatic 6/24/72 h | water 6/24/72 h | arrival (h, half of 72 h): T / u / Φ / q | order |
|---|---|---|---|---|---|---|
| `no_temp_advection` | 0.080 / 0.201 / 0.367 | 0.629 / 0.606 / 0.686 | 0.053 / 0.134 / 0.263 | 0.237 / 0.330 / 0.567 | 24 / 6 / 24 / 18 | u → q → T = Φ |
| `no_adiabatic` | – / – / – | – / – / – | – / – / – | – / – / – | – / – / – / – | – |
| `no_density` | 0.624 / – / – | 3.978 / – / – | 1.159 / – / – | 1.830 / – / – | – / – / – / – | – |
| `weak_density` | 0.024 / 0.028 / 0.056 | 0.169 / 0.173 / 0.190 | 0.103 / 0.035 / 0.065 | 0.061 / 0.074 / 0.139 | 18 / 6 / 6 / 18 | u = Φ → T = q |
| `no_virtual` | 0.008 / 0.014 / 0.027 | 0.085 / 0.064 / 0.081 | 0.042 / 0.020 / 0.031 | 0.067 / 0.065 / 0.116 | 24 / 6 / 6 / 6 | q = u = Φ → T |
| `no_moisture` | 0.004 / 0.027 / 0.110 | 0.055 / 0.143 / 0.280 | 0.005 / 0.025 / 0.096 | 0.244 / 0.476 / 0.688 | 42 / 24 / 42 / 12 | q → u → T = Φ |
| `no_pgf` | – / – / – | – / – / – | – / – / – | – / – / – | – / – / – / – | – |
| `no_advection` | 0.087 / 0.233 / 0.381 | 0.648 / 0.774 / 0.810 | 0.120 / 0.282 / 0.456 | 0.201 / 0.352 / 0.607 | 18 / 6 / 18 / 18 | u → T = q = Φ |
| `no_coriolis` | 0.384 / 0.885 / 1.159 | 2.582 / 2.924 / 2.645 | 1.786 / 1.650 / 1.056 | 0.987 / 2.198 / 3.442 | 18 / 6 / 6 / 18 | u = Φ → T = q |
| `no_vertical` | 0.060 / 0.137 / 0.262 | 0.501 / 0.472 / 0.576 | 0.039 / 0.098 / 0.225 | 0.257 / 0.381 / 0.604 | 24 / 6 / 30 / 12 | u → q → T → Φ |
| `no_drag` | 0.004 / 0.017 / 0.085 | 0.069 / 0.130 / 0.313 | 0.014 / 0.055 / 0.157 | 0.020 / 0.060 / 0.255 | 48 / 36 / 36 / 48 | u = Φ → T = q |
| `no_carry` | 0.067 / 0.067 / 0.067 | 0.093 / 0.093 / 0.093 | 0.030 / 0.030 / 0.030 | 0.096 / 0.096 / 0.096 | 6 / 6 / 6 / 6 | T = q = u = Φ |
| `no_heating` | 0.045 / 0.137 / 0.349 | 0.424 / 0.495 / 0.654 | 0.029 / 0.074 / 0.170 | 0.176 / 0.287 / 0.515 | 36 / 6 / 30 / 18 | u → q → Φ → T |
| `no_moistening` | 0.007 / – / – | 0.086 / – / – | 0.010 / – / – | 0.400 / – / – | – / – / – / – | – |
| `no_momentum` | 0.011 / 0.041 / 0.117 | 0.166 / 0.171 / 0.308 | 0.015 / 0.073 / 0.229 | 0.054 / 0.100 / 0.245 | 42 / 6 / 42 / 36 | u → q → T = Φ |
| `no_surface_pressure` | 0.003 / 0.014 / 0.091 | 0.074 / 0.071 / 0.254 | 0.032 / 0.112 / 0.345 | 0.022 / 0.039 / 0.150 | 54 / 48 / 42 / 48 | Φ → q = u → T |
| `q_to_heating` | 0.027 / 0.088 / 0.236 | 0.214 / 0.246 / 0.417 | 0.029 / 0.083 / 0.211 | 0.143 / 0.256 / 0.496 | 36 / 6 / 36 / 24 | u → q → T = Φ |
| `t_to_moistening` | 0.008 / 0.029 / 0.091 | 0.053 / 0.081 / 0.204 | 0.013 / 0.050 / 0.111 | 0.337 / 0.480 / 0.705 | 42 / 42 / 30 / 12 | q → Φ → T = u |
| `sun_off` | 0.019 / 0.026 / 0.069 | 0.027 / 0.048 / 0.096 | 0.009 / 0.030 / 0.082 | 0.060 / 0.086 / 0.156 | 30 / 24 / 36 / 24 | q = u → T → Φ |
| `sun_to_heating` | 0.009 / 0.026 / 0.064 | 0.022 / 0.046 / 0.089 | 0.007 / 0.021 / 0.048 | 0.011 / 0.044 / 0.114 | 36 / 24 / 36 / 36 | u → T = q = Φ |
| `ocean_column` | 0.014 / 0.045 / 0.129 | 0.100 / 0.125 / 0.264 | 0.019 / 0.064 / 0.179 | 0.073 / 0.114 / 0.230 | 36 / 30 / 42 / 30 | q = u → T → Φ |

## The sun (2018)

| intervention | Δt2m day 6/24/72 h (K) | Δt2m night | r(Δt2m, sunlight) |
|---|---|---|---|
| `sun_off` | -1.13 / -0.94 / -1.97 | -0.04 / -0.50 / -1.33 | -0.56 / -0.33 / -0.44 |
| `sun_to_heating` | -0.23 / -0.56 / -1.50 | -0.04 / -0.52 / -1.25 | -0.63 / -0.59 / -0.58 |
| `ocean_column` | -0.02 / +0.18 / +0.94 | +0.04 / +0.02 / +0.58 | -0.13 / -0.02 / -0.07 |
| `q_to_heating` | +0.03 / +0.19 / +0.73 | +0.07 / +0.30 / +0.89 | -0.16 / -0.26 / -0.19 |
| `no_heating` | -0.33 / -0.64 / -2.15 | +0.06 / -0.81 / -2.21 | -0.33 / +0.09 / +0.04 |
| `sun_off+no_heating` | -1.24 / -1.03 / -2.50 | +0.06 / -0.81 / -2.20 | -0.53 / +0.04 / +0.03 |

## The sun (2020)

| intervention | Δt2m day 6/24/72 h (K) | Δt2m night | r(Δt2m, sunlight) |
|---|---|---|---|
| `sun_off` | -1.13 / -0.95 / -1.96 | -0.04 / -0.50 / -1.34 | -0.56 / -0.33 / -0.44 |
| `sun_to_heating` | -0.23 / -0.56 / -1.50 | -0.04 / -0.53 / -1.26 | -0.63 / -0.59 / -0.58 |
| `ocean_column` | -0.01 / +0.21 / +0.95 | +0.05 / +0.03 / +0.64 | -0.15 / -0.06 / -0.07 |
| `q_to_heating` | +0.02 / +0.16 / +0.57 | +0.07 / +0.29 / +0.85 | -0.18 / -0.26 / -0.16 |
| `no_heating` | -0.34 / -0.67 / -2.12 | +0.05 / -0.88 / -2.64 | -0.33 / +0.11 / +0.11 |
| `sun_off+no_heating` | -1.25 / -1.07 / -2.48 | +0.05 / -0.88 / -2.63 | -0.53 / +0.06 / +0.10 |

## Pairs: loops of the diagram (2018)

interaction = change(A+B) - change(A) - change(B); eta = |D_AB - D_A - D_B| / (|D_A| + |D_B|) (0: independent paths); rho(A given B) = share of A's effect left once B is cut (0: B mediates A).

| pair (A + B) | interaction, all-MSE 6/24/72 h | eta T / u / Φ / q at 72 h | rho(A given B) T / u / Φ / q at 72 h | rho(B given A) at 72 h |
|---|---|---|---|---|
| `no_temp_advection` + `weak_density` | -2% / +1% / -1% | 0.19 / 0.31 / 0.15 / 0.21 | 1.00 / 0.98 / 0.95 / 0.97 | 1.28 / 1.00 / 0.76 / 0.47 |
| `no_moisture` + `no_virtual` | -1% / -0% / +2% | 0.16 / 0.19 / 0.16 / 0.15 | 0.89 / 0.86 / 0.85 / 0.94 | 0.59 / 0.65 / 0.70 / 0.19 |
| `no_moisture` + `q_to_heating` | -3% / -9% / +0% | 0.31 / 0.39 / 0.27 / 0.43 | 0.14 / 0.16 / 0.28 / 0.62 | 0.84 / 0.68 / 0.88 / 0.28 |
| `t_to_moistening` + `q_to_heating` | -4% / -2% / +5% | 0.25 / 0.29 / 0.18 / 0.28 | 0.23 / 0.28 / 0.76 / 0.73 | 0.84 / 0.85 / 0.88 / 0.69 |
| `sun_off` + `no_heating` | -1% / -1% / -0% | 0.16 / 0.13 / 0.20 / 0.17 | 0.16 / 0.44 / 0.80 / 0.61 | 0.98 / 0.99 / 1.03 / 0.98 |
| `sun_off` + `q_to_heating` | -1% / +0% / +2% | 0.07 / 0.15 / 0.05 / 0.15 | 0.99 / 0.79 / 0.99 / 0.82 | 1.03 / 1.01 / 1.04 / 1.00 |
| `sun_off` + `ocean_column` | -0% / -1% / -2% | 0.16 / 0.22 / 0.11 / 0.25 | 0.95 / 0.54 / 0.92 / 0.80 | 1.01 / 1.02 / 1.01 / 1.03 |

## Pairs: loops of the diagram (2020)

interaction = change(A+B) - change(A) - change(B); eta = |D_AB - D_A - D_B| / (|D_A| + |D_B|) (0: independent paths); rho(A given B) = share of A's effect left once B is cut (0: B mediates A).

| pair (A + B) | interaction, all-MSE 6/24/72 h | eta T / u / Φ / q at 72 h | rho(A given B) T / u / Φ / q at 72 h | rho(B given A) at 72 h |
|---|---|---|---|---|
| `no_temp_advection` + `weak_density` | -2% / +2% / -2% | 0.19 / 0.32 / 0.16 / 0.21 | 1.01 / 0.98 / 0.95 / 0.98 | 1.35 / 1.06 / 0.79 / 0.53 |
| `no_moisture` + `no_virtual` | -1% / -1% / +2% | 0.15 / 0.18 / 0.16 / 0.14 | 0.90 / 0.87 / 0.86 / 0.94 | 0.62 / 0.71 / 0.70 / 0.22 |
| `no_moisture` + `q_to_heating` | -3% / -10% / -1% | 0.31 / 0.38 / 0.27 / 0.42 | 0.14 / 0.17 / 0.29 / 0.65 | 0.84 / 0.70 / 0.88 / 0.29 |
| `t_to_moistening` + `q_to_heating` | -4% / -2% / +4% | 0.25 / 0.28 / 0.18 / 0.27 | 0.23 / 0.30 / 0.76 / 0.74 | 0.84 / 0.86 / 0.88 / 0.70 |
| `sun_off` + `no_heating` | -1% / -1% / -1% | 0.16 / 0.12 / 0.19 / 0.17 | 0.16 / 0.50 / 0.80 / 0.63 | 0.98 / 0.99 / 1.03 / 0.97 |
| `sun_off` + `q_to_heating` | -0% / +0% / +2% | 0.07 / 0.14 / 0.05 / 0.14 | 1.00 / 0.84 / 1.00 / 0.84 | 1.03 / 1.01 / 1.04 / 1.00 |
| `sun_off` + `ocean_column` | -0% / -1% / -2% | 0.16 / 0.22 / 0.11 / 0.25 | 0.96 / 0.55 / 0.91 / 0.81 | 1.01 / 1.01 / 1.00 / 1.03 |

## Invariance 2018 vs 2020

Sign agreement on the 302 per-variable changes significant in both years (24 and 72 h): 100%. Rank correlation of the 72 h effect sizes over 24 interventions: 0.95.

## Expectations (pre-registered)

| | expectation | verdict | numbers |
|---|---|---|---|
| E1 | do(TISR = 0) cools day points more than night points at 6 h, and the 2 m temperature response correlates with the withheld energy (r < -0.3) | supported | 6 h: day -1.13 K, night -0.04 K, r = -0.56 |
| E2 | the column heating carries the sun's effect on temperature by 72 h: rho(sun_off given no_heating) < 0.5 | supported | rho(sun given no heating) on T at 6 h 0.81, 24 h 0.29, 72 h 0.16 |
| E3 | the latent-heat loop is non-additive: eta(no_moisture + q_to_heating) >= 0.1 at 72 h on temperature or water, larger than at 6 h | supported | eta 6 → 72 h: thermodynamics 0.11 → 0.31, water 0.25 → 0.43 |
| E4 | effects propagate: the response grows from 6 h to 72 h for >= 80% of (intervention, group) | supported | 90% of (intervention, group) grow from 6 to 72 h |
| E5 | the transports that N0 is better without (no_vertical, no_moisture at 72 h) hurt N1 | supported | no_vertical: N0 -20%, N1 +16%; no_moisture: N0 -26%, N1 +1% |
| E6 | invariance: >= 80% sign agreement 2018 vs 2020 on the significant per-variable changes at 24 and 72 h, and rank correlation of the 72 h effect sizes >= 0.8 | supported | sign agreement 100% of 302 significant changes; rank r = 0.95 over 24 |
