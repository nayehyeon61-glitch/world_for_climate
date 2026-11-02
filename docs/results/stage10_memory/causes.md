# Stage 10: the memory model (D-noactor) under the stage 9 interventions, 2020, against N1

Model: `runs/stage10_Dnoactor/checkpoint.pt`; 8 initialisations per year (memory, N1), 72 h, the same dates in N0. Changes: channel-mean normalised MSE vs the intact model (+ = worse without the arrow). Blow-ups: runs that diverged by 72 h (per year).

## The arrows

| intervention | arrow | memory 6 h | memory 24 h | memory 72 h | N1 6 h | N1 24 h | N1 72 h | blow-ups |
|---|---|---|---|---|---|---|---|---|
| `no_temp_advection` | motion -> thermodynamics (u, omega advect T) | +64% | +43% | -11% | +104% | +73% | +31% | 0/0 |
| `no_adiabatic` | motion -> thermodynamics (kappa T omega / p) | blown | blown | blown | blown | blown | blown | 8/8 |
| `no_density` | thermodynamics -> motion (T' in the pressure gradient: density) | +4518% | blown | blown | +5981% | blown | blown | 8/8 |
| `weak_density` | thermodynamics -> motion, T' coupling 10% weaker (soft: the full cut blows up) | +7% | +4% | -3% | +11% | +7% | +1% | 0/0 |
| `no_virtual` | water -> hydrostatic balance (T_v: buoyancy of water vapour) | +1% | -0% | -2% | +3% | +1% | -2% | 0/0 |
| `no_moisture` | motion -> water (u, omega advect q) | +6% | +8% | -4% | +9% | +19% | +2% | 0/0 |
| `no_pgf` | hydrostatic balance -> motion (the whole pressure gradient: Phi, Phi_b) | blown | blown | blown | blown | blown | blown | 8/8 |
| `no_advection` | motion -> motion (horizontal advection of momentum, T and Phi_b) | +92% | +93% | +11% | +133% | +146% | +63% | 0/0 |
| `no_coriolis` | rotation -> motion (Coriolis) | +2648% | +1557% | +331% | +2880% | +2807% | +1668% | 5/7 |
| `no_vertical` | motion -> all (vertical advection) | +42% | +23% | -13% | +68% | +45% | +20% | 0/0 |
| `no_drag` | surface -> motion (boundary-layer friction) | +2% | +6% | +14% | +2% | +5% | +17% | 0/0 |
| `no_carry` | initial residual (unresolved scales, offsets) | +0% | -2% | -1% | +1% | -2% | -1% | 0/0 |
| `no_heating` | radiation + latent heating -> thermodynamics (column Q) | +37% | +34% | -13% | +60% | +56% | +36% | 0/0 |
| `no_moistening` | water sources and sinks (column S) | -6% | blown | blown | +21% | blown | blown | 8/8 |
| `no_momentum` | column momentum forcing -> motion | +9% | +13% | +48% | +11% | +9% | +18% | 0/0 |
| `no_surface_pressure` | column -> surface pressure (P_b) | -1% | -5% | -29% | +3% | +1% | +0% | 0/0 |
| `q_to_heating` | water -> thermodynamics (latent heating) and water -> radiation (moisture) | +7% | -1% | -37% | +21% | +16% | +5% | 0/0 |
| `t_to_moistening` | thermodynamics -> water (saturation, condensation) | +16% | +7% | -31% | +14% | +21% | +16% | 0/0 |
| `sun_off` | sun -> radiation: do(TISR = 0) | +1% | +0% | +2% | +0% | +1% | +1% | 0/0 |
| `sun_to_heating` | radiation -> thermodynamics (heating and cooling), the sun's path only | +0% | -1% | -3% | +0% | +0% | -0% | 0/0 |
| `ocean_column` | surface (land / ocean) -> radiation and column physics | +2% | -1% | -15% | +3% | +3% | -0% | 0/0 |

Arrows without a path in N1:

- surface <-> water (precipitation, evaporation): evaporation E only diagnoses precipitation (P = E - int S dp / g); no tendency reads E, and nothing stores surface water or heat

## Per variable at 72 h, memory

| intervention | z500 | t850 | q700 | u850 | t2m | msl | u10 | tp |
|---|---|---|---|---|---|---|---|---|
| `no_temp_advection` | +8% | +15% | -10% | +1% | -22% | -4% | -4% | -7% |
| `no_adiabatic` | blown | blown | blown | blown | blown | blown | blown | blown |
| `no_density` | blown | blown | blown | blown | blown | blown | blown | blown |
| `weak_density` | +8% | +4% | -2% | -3% | +3% | +6% | -3% | -1% |
| `no_virtual` | +2% | +1% | -3% | -1% | +1% | +2% | -2% | -1% |
| `no_moisture` | +13% | +5% | -17% | +3% | +2% | +12% | -1% | -1% |
| `no_pgf` | blown | blown | blown | blown | blown | blown | blown | blown |
| `no_advection` | +29% | +15% | +3% | +13% | +2% | +40% | +7% | -6% |
| `no_coriolis` | +37% | +2% | +82% | +105% | -13% | +67% | +42% | +11% |
| `no_vertical` | +15% | +10% | -15% | +3% | -21% | +5% | -4% | -3% |
| `no_drag` | +13% | +7% | +4% | +18% | +5% | +17% | +23% | +4% |
| `no_carry` | -0% | -0% | -1% | -1% | -9% | +3% | -1% | +0% |
| `no_heating` | -55% | -56% | -19% | -5% | -60% | -11% | -2% | -8% |
| `no_moistening` | blown | blown | blown | blown | blown | blown | blown | blown |
| `no_momentum` | +42% | +19% | +3% | +23% | +15% | +63% | +44% | +8% |
| `no_surface_pressure` | -56% | -22% | -7% | -24% | -18% | +2% | -2% | -7% |
| `q_to_heating` | -36% | -54% | -20% | +2% | -54% | +4% | +1% | -10% |
| `t_to_moistening` | -47% | -34% | -14% | -16% | -29% | -44% | -7% | -6% |
| `sun_off` | +17% | +6% | -4% | +2% | +7% | +17% | -1% | +1% |
| `sun_to_heating` | +4% | +0% | -3% | +0% | +2% | -2% | -0% | -0% |
| `ocean_column` | -31% | -10% | -1% | -13% | -15% | -35% | -3% | -2% |
| `no_temp_advection+weak_density` | +15% | +21% | -11% | +1% | -19% | -1% | -4% | -7% |
| `no_moisture+no_virtual` | +16% | +7% | -17% | +3% | +3% | +13% | -2% | -1% |
| `no_moisture+q_to_heating` | -31% | -54% | -22% | +3% | -53% | +11% | +1% | -9% |
| `t_to_moistening+q_to_heating` | -57% | -57% | -9% | -10% | -55% | -30% | -4% | -10% |
| `sun_off+no_heating` | -53% | -56% | -20% | -4% | -59% | -4% | -3% | -8% |
| `sun_off+q_to_heating` | -21% | -45% | -22% | +4% | -53% | +18% | +3% | -10% |
| `sun_off+ocean_column` | -20% | -8% | -5% | -13% | -14% | -26% | -4% | -2% |

## Per variable at 72 h, N1

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

## Propagation (memory): RMS response vs the intact model, normalised units

| intervention | thermodynamics 6/24/72 h | momentum 6/24/72 h | hydrostatic 6/24/72 h | water 6/24/72 h | arrival (h, half of 72 h): T / u / Φ / q | order |
|---|---|---|---|---|---|---|
| `no_temp_advection` | 0.086 / 0.229 / 0.469 | 0.663 / 0.661 / 0.857 | 0.055 / 0.139 / 0.281 | 0.238 / 0.338 / 0.728 | 30 / 6 / 30 / 30 | u → T = q = Φ |
| `no_adiabatic` | – / – / – | – / – / – | – / – / – | – / – / – | – / – / – / – | – |
| `no_density` | 0.619 / – / – | 4.057 / – / – | 1.226 / – / – | 2.119 / – / – | – / – / – / – | – |
| `weak_density` | 0.025 / 0.032 / 0.079 | 0.168 / 0.180 / 0.256 | 0.103 / 0.042 / 0.107 | 0.079 / 0.082 / 0.198 | 36 / 6 / 6 / 42 | u = Φ → T → q |
| `no_virtual` | 0.009 / 0.013 / 0.028 | 0.079 / 0.056 / 0.080 | 0.042 / 0.020 / 0.037 | 0.047 / 0.044 / 0.095 | 30 / 6 / 6 / 36 | u = Φ → T → q |
| `no_moisture` | 0.004 / 0.031 / 0.158 | 0.059 / 0.138 / 0.403 | 0.005 / 0.030 / 0.148 | 0.245 / 0.478 / 0.859 | 48 / 42 / 48 / 24 | q → u → T = Φ |
| `no_pgf` | – / – / – | – / – / – | – / – / – | – / – / – | – / – / – / – | – |
| `no_advection` | 0.096 / 0.271 / 0.503 | 0.690 / 0.809 / 0.927 | 0.123 / 0.289 / 0.523 | 0.305 / 0.409 / 0.772 | 24 / 6 / 24 / 24 | u → T = q = Φ |
| `no_coriolis` | 0.416 / 0.891 / 1.213 | 2.955 / 2.628 / 2.308 | 1.742 / 1.469 / 1.256 | 1.767 / 1.815 / 2.094 | 12 / 6 / 6 / 6 | q = u = Φ → T |
| `no_vertical` | 0.063 / 0.154 / 0.349 | 0.502 / 0.488 / 0.719 | 0.041 / 0.101 / 0.263 | 0.237 / 0.375 / 0.781 | 30 / 6 / 36 / 30 | u → T = q → Φ |
| `no_drag` | 0.004 / 0.019 / 0.101 | 0.067 / 0.128 / 0.342 | 0.014 / 0.053 / 0.163 | 0.022 / 0.058 / 0.300 | 54 / 42 / 42 / 54 | u = Φ → T = q |
| `no_carry` | 0.067 / 0.067 / 0.067 | 0.093 / 0.093 / 0.093 | 0.028 / 0.028 / 0.028 | 0.101 / 0.101 / 0.101 | 6 / 6 / 6 / 6 | T = q = u = Φ |
| `no_heating` | 0.071 / 0.246 / 0.715 | 0.515 / 0.601 / 1.050 | 0.047 / 0.163 / 0.551 | 0.143 / 0.275 / 0.779 | 36 / 12 / 42 / 42 | u → T → q = Φ |
| `no_moistening` | 0.015 / – / – | 0.122 / – / – | 0.020 / – / – | 0.631 / – / – | – / – / – / – | – |
| `no_momentum` | 0.012 / 0.055 / 0.247 | 0.198 / 0.227 / 0.672 | 0.021 / 0.105 / 0.457 | 0.091 / 0.144 / 0.596 | 48 / 48 / 48 / 54 | T = u = Φ → q |
| `no_surface_pressure` | 0.006 / 0.035 / 0.264 | 0.103 / 0.134 / 0.652 | 0.059 / 0.232 / 0.960 | 0.053 / 0.075 / 0.459 | 54 / 48 / 48 / 54 | u = Φ → T = q |
| `q_to_heating` | 0.046 / 0.168 / 0.493 | 0.220 / 0.298 / 0.684 | 0.046 / 0.152 / 0.422 | 0.102 / 0.245 / 0.735 | 36 / 30 / 36 / 42 | u → T = Φ → q |
| `t_to_moistening` | 0.010 / 0.040 / 0.251 | 0.060 / 0.080 / 0.478 | 0.019 / 0.077 / 0.425 | 0.436 / 0.601 / 1.054 | 54 / 54 / 54 / 18 | q → T = u = Φ |
| `sun_off` | 0.024 / 0.046 / 0.145 | 0.031 / 0.053 / 0.176 | 0.017 / 0.064 / 0.221 | 0.076 / 0.105 / 0.225 | 36 / 48 / 42 / 30 | q → T → Φ → u |
| `sun_to_heating` | 0.012 / 0.035 / 0.094 | 0.023 / 0.045 / 0.141 | 0.009 / 0.028 / 0.068 | 0.010 / 0.033 / 0.137 | 36 / 48 / 36 / 54 | T = Φ → u → q |
| `ocean_column` | 0.020 / 0.078 / 0.260 | 0.119 / 0.161 / 0.444 | 0.025 / 0.098 / 0.380 | 0.092 / 0.134 / 0.386 | 42 / 42 / 48 / 42 | T = q = u → Φ |

## The sun (memory)

| intervention | Δt2m day 6/24/72 h (K) | Δt2m night | r(Δt2m, sunlight) |
|---|---|---|---|
| `sun_off` | -1.41 / -1.51 / -4.01 | -0.07 / -0.92 / -2.97 | -0.60 / -0.39 / -0.43 |
| `sun_to_heating` | -0.33 / -0.82 / -2.32 | -0.06 / -0.75 / -1.91 | -0.64 / -0.59 / -0.55 |
| `ocean_column` | +0.04 / +0.51 / +2.53 | +0.08 / +0.30 / +2.39 | -0.11 / -0.14 / -0.18 |
| `q_to_heating` | -0.12 / -0.01 / +2.57 | +0.04 / +0.45 / +4.05 | -0.26 / -0.61 / -0.59 |
| `no_heating` | -0.77 / -1.67 / -1.16 | -0.18 / -1.40 / +0.50 | -0.42 / -0.56 / -0.56 |
| `sun_off+no_heating` | -1.81 / -2.15 / -1.60 | -0.18 / -1.43 / +0.44 | -0.60 / -0.57 / -0.56 |

## The sun (N1)

| intervention | Δt2m day 6/24/72 h (K) | Δt2m night | r(Δt2m, sunlight) |
|---|---|---|---|
| `sun_off` | -1.13 / -0.95 / -1.96 | -0.04 / -0.50 / -1.34 | -0.56 / -0.33 / -0.44 |
| `sun_to_heating` | -0.23 / -0.56 / -1.50 | -0.04 / -0.53 / -1.26 | -0.63 / -0.59 / -0.58 |
| `ocean_column` | -0.01 / +0.21 / +0.95 | +0.05 / +0.03 / +0.64 | -0.15 / -0.06 / -0.07 |
| `q_to_heating` | +0.02 / +0.16 / +0.57 | +0.07 / +0.29 / +0.85 | -0.18 / -0.26 / -0.16 |
| `no_heating` | -0.34 / -0.67 / -2.12 | +0.05 / -0.88 / -2.64 | -0.33 / +0.11 / +0.11 |
| `sun_off+no_heating` | -1.25 / -1.07 / -2.48 | +0.05 / -0.88 / -2.63 | -0.53 / +0.06 / +0.10 |

## Pairs: loops of the diagram (memory)

interaction = change(A+B) - change(A) - change(B); eta = |D_AB - D_A - D_B| / (|D_A| + |D_B|) (0: independent paths); rho(A given B) = share of A's effect left once B is cut (0: B mediates A).

| pair (A + B) | interaction, all-MSE 6/24/72 h | eta T / u / Φ / q at 72 h | rho(A given B) T / u / Φ / q at 72 h | rho(B given A) at 72 h |
|---|---|---|---|---|
| `no_temp_advection` + `weak_density` | -1% / +1% / +4% | 0.18 / 0.31 / 0.17 / 0.23 | 1.01 / 0.99 / 0.98 / 0.99 | 1.08 / 0.96 / 0.76 / 0.53 |
| `no_moisture` + `no_virtual` | -1% / +1% / +2% | 0.12 / 0.14 / 0.10 / 0.10 | 0.95 / 0.93 / 0.94 / 0.96 | 0.67 / 0.67 / 0.83 / 0.32 |
| `no_moisture` + `q_to_heating` | -2% / -2% / +7% | 0.24 / 0.36 / 0.20 / 0.43 | 0.13 / 0.15 / 0.39 / 0.53 | 0.96 / 0.76 / 1.01 / 0.47 |
| `t_to_moistening` + `q_to_heating` | -3% / +3% / +27% | 0.30 / 0.36 / 0.27 / 0.29 | 0.23 / 0.32 / 0.60 / 0.65 | 0.68 / 0.72 / 0.69 / 0.59 |
| `sun_off` + `no_heating` | -0% / -0% / -1% | 0.17 / 0.14 / 0.15 / 0.17 | 0.12 / 0.41 / 0.72 / 0.57 | 1.01 / 0.98 / 1.05 / 0.97 |
| `sun_off` + `q_to_heating` | -0% / +1% / +2% | 0.09 / 0.17 / 0.07 / 0.14 | 0.71 / 0.61 / 0.87 / 0.78 | 1.01 / 0.97 / 1.02 / 0.98 |
| `sun_off` + `ocean_column` | -0% / -1% / -4% | 0.14 / 0.20 / 0.08 / 0.19 | 0.97 / 0.73 / 0.93 / 0.87 | 1.03 / 1.00 / 1.03 / 0.99 |

## Pairs: loops of the diagram (N1)

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

## Invariance memory vs N1

Sign agreement on the 272 per-variable changes significant in both years (24 and 72 h): 76%. Rank correlation of the 72 h effect sizes over 24 interventions: 0.35.

## Expectations (pre-registered)

| | expectation | verdict | numbers |
|---|---|---|---|
| E1 | do(TISR = 0) cools day points more than night points at 6 h, and the 2 m temperature response correlates with the withheld energy (r < -0.3) | supported | 6 h: day -1.41 K, night -0.07 K, r = -0.60 |
| E2 | the column heating carries the sun's effect on temperature by 72 h: rho(sun_off given no_heating) < 0.5 | supported | rho(sun given no heating) on T at 6 h 0.71, 24 h 0.20, 72 h 0.12 |
| E3 | the latent-heat loop is non-additive: eta(no_moisture + q_to_heating) >= 0.1 at 72 h on temperature or water, larger than at 6 h | supported | eta 6 → 72 h: thermodynamics 0.08 → 0.24, water 0.25 → 0.43 |
| E4 | effects propagate: the response grows from 6 h to 72 h for >= 80% of (intervention, group) | supported | 94% of (intervention, group) grow from 6 to 72 h |
| E6 | invariance: >= 80% sign agreement 2018 vs 2020 on the significant per-variable changes at 24 and 72 h, and rank correlation of the 72 h effect sizes >= 0.8 | not supported | sign agreement 76% of 272 significant changes; rank r = 0.35 over 24 |
