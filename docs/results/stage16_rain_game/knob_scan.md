# Stage 16 knob scan: constant settings, 15-day free runs of N1 on 8 training windows

a = +-1 is the end of a knob's range (multiplier 1 + 0.5 a; diffusion time x 2^a, a = +1 = weaker diffusion). Alive days count 15 for runs that never blew up. Daily distance = the game's 5-day smoothed zonal-mean distance to ERA5 on day 8 (normalised units), mean over the runs still alive. The reading is in README_KO section 16.

| setting | blown | alive days mean | day 8 distance | return |
|---|---|---|---|---|
| base | 8/8 | 8.0 | 0.168 | -42.1 |
| diffusion-1 Q_upper-1 drag+0.5 | 4/8 | 13.1 | 0.141 | -28.0 |
| diffusion-1 Q_upper-1 | 4/8 | 13.0 | 0.149 | -28.9 |
| cem1 best | 3/8 | 12.4 | 0.109 | -25.1 |
| Q_upper-1 | 6/8 | 11.8 | 0.153 | -33.8 |
| Q_lower-1 | 8/8 | 10.0 | 0.168 | -39.3 |
| Q_upper-0.5 | 8/8 | 9.8 | 0.153 | -38.8 |
| diffusion-1 (tau/2, stronger) | 8/8 | 9.1 | 0.159 | -40.0 |
| S_upper+0.2 | 8/8 | 8.8 | 0.159 | -40.9 |
| S_upper+0.05 | 8/8 | 8.2 | 0.163 | -41.6 |
| Q_pbl-1 | 8/8 | 8.2 | 0.162 | -41.6 |
| S_pbl-0.2 | 8/8 | 8.1 | 0.160 | -41.9 |
| drag+1 | 8/8 | 7.9 | 0.144 | -42.1 |
| F-1 | 8/8 | 7.9 | 0.174 | -42.8 |
| S_upper-0.05 | 8/8 | 7.8 | 0.170 | -42.7 |
| Q_strat-1 | 8/8 | 7.5 | 0.117 | -41.2 |
| S_upper-0.2 | 8/8 | 7.5 | 0.179 | -44.0 |
| Q_strat+1 | 8/8 | 7.5 | 0.219 | -46.0 |
| S_lower-0.2 | 8/8 | 7.2 | 0.171 | -43.6 |
| diffusion+1 (tau x2, weaker) | 8/8 | 6.9 | 0.164 | -44.4 |
| S_upper+1 | 8/8 | 5.1 | 0.671 | -56.9 |
| Q_upper+1 | 8/8 | 1.6 | - | -55.7 |
| combo diffusion+1 (weaker) S_upper-1 Q_strat-1 | 8/8 | 0.5 | - | -57.5 |
| S_upper-1 | 8/8 | 0.5 | - | -57.5 |
| S_all-1 | 8/8 | 0.4 | - | -57.8 |

cem1 best = the best single candidate of the first CEM run (spread floor 0.05, stopped at iteration 9): Q_strat -0.65, Q_upper -1.0,
Q_lower -0.17, Q_pbl +0.37, S_upper +0.12, S_lower -0.02, S_pbl +0.12, F +0.12, drag +0.42, diffusion -0.14.
