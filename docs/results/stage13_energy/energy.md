# Stage 13: the column's energy input and the blow-ups

E1 bounds the learned column's energy input at run time (physics_world_model.energy_bound): the moist static energy flux of every column is clamped to +-500 W/m^2 (the excess heating removed uniformly per unit mass) and the momentum forcing may not do work on the wind. No retraining. Design and rules B1-B4: README_KO §13.

## memory: free runs under the bounds (2020, 10 initialisations, 120 h)

Blow-ups per start kind and setting; the error ratio against the unbounded model over the initialisations finite in both (GraphCast-weighted all-point MSE, normalised units); the energy terms are means over the run (W/m^2).

| start | setting | blow-ups | error ratio vs none (24 / 72 / 120 h) | heating | moistening | work | largest flux | clamped |
|---|---|---|---|---|---|---|---|---|
| none | none | 5/10 | reference | nan | nan | nan | nan | 0.0% |
| none | flux500 | 2/10 | 0.982 / 0.875 / 0.660 (n=4) | nan | nan | nan | nan | 23.6% |
| none | flux250 | 2/10 | 0.975 / 0.790 / 0.802 (n=4) | nan | nan | nan | nan | 50.4% |
| none | nowork | 6/10 | 0.997 / 0.976 / 0.969 (n=4) | nan | nan | nan | nan | 0.0% |
| none | flux500_nowork | 2/10 | 0.979 / 0.852 / 0.631 (n=4) | nan | nan | nan | nan | 23.1% |
| region | none | 2/10 | reference | nan | nan | nan | nan | 0.0% |
| region | flux500 | 0/10 | 1.022 / 0.957 / 0.826 (n=8) | 18.8 | 4.0 | -2.00 | 2380 | 23.1% |
| region | flux250 | 1/10 | 1.038 / 0.932 / 0.943 (n=8) | 27.4 | 4.3 | -1.78 | 2650 | 51.3% |
| region | nowork | 1/10 | 0.998 / 0.989 / 0.964 (n=8) | nan | nan | nan | nan | 0.0% |
| region | flux500_nowork | 0/10 | 1.019 / 0.947 / 0.808 (n=8) | 20.4 | 4.1 | -2.00 | 2349 | 22.7% |

The energy terms over the four steps before a blow-up, against the same leads of the runs that stayed finite:

| run | term | before blow-up | finite runs, same leads | ratio |
|---|---|---|---|---|
| none:none | heating c_p∫Q (global mean) | -107 | -61.9 | 1.7 |
| none:none | moistening L_v∫S (global mean) | -15.1 | -9.63 | 1.6 |
| none:none | work ∫F·u (global mean) | -5.1 | -4.33 | 1.2 |
| none:none | largest |flux| of a column | 6.03e+03 | 3.78e+03 | 1.6 |
| none:none | largest work of a column | 75.2 | 49.8 | 1.5 |
| none:none | clamped fraction | 0 | 0 | 0.0 |
| none:flux500 | heating c_p∫Q (global mean) | -110 | -74.6 | 1.5 |
| none:flux500 | moistening L_v∫S (global mean) | -12.1 | -7.34 | 1.6 |
| none:flux500 | work ∫F·u (global mean) | -3.5 | -3.83 | 0.9 |
| none:flux500 | largest |flux| of a column | 5.71e+03 | 3.7e+03 | 1.5 |
| none:flux500 | largest work of a column | 93.5 | 18.1 | 5.2 |
| none:flux500 | clamped fraction | 0.309 | 0.284 | 1.1 |
| none:flux250 | heating c_p∫Q (global mean) | -89.4 | -47.5 | 1.9 |
| none:flux250 | moistening L_v∫S (global mean) | -7.52 | -2.93 | 2.6 |
| none:flux250 | work ∫F·u (global mean) | -2.69 | -2.81 | 1.0 |
| none:flux250 | largest |flux| of a column | 3.99e+03 | 3.49e+03 | 1.1 |
| none:flux250 | largest work of a column | 43.9 | 19.3 | 2.3 |
| none:flux250 | clamped fraction | 0.565 | 0.532 | 1.1 |
| none:nowork | heating c_p∫Q (global mean) | -105 | -66.6 | 1.6 |
| none:nowork | moistening L_v∫S (global mean) | -14.8 | -10.2 | 1.5 |
| none:nowork | work ∫F·u (global mean) | -4.79 | -4.87 | 1.0 |
| none:nowork | largest |flux| of a column | 6.25e+03 | 3.32e+03 | 1.9 |
| none:nowork | largest work of a column | 70.5 | 16.7 | 4.2 |
| none:nowork | clamped fraction | 0 | 0 | 0.0 |
| none:flux500_nowork | heating c_p∫Q (global mean) | -105 | -71.3 | 1.5 |
| none:flux500_nowork | moistening L_v∫S (global mean) | -11.3 | -7.07 | 1.6 |
| none:flux500_nowork | work ∫F·u (global mean) | -3.45 | -3.76 | 0.9 |
| none:flux500_nowork | largest |flux| of a column | 5.23e+03 | 3.61e+03 | 1.4 |
| none:flux500_nowork | largest work of a column | 78.7 | 16.5 | 4.8 |
| none:flux500_nowork | clamped fraction | 0.3 | 0.277 | 1.1 |
| region:none | heating c_p∫Q (global mean) | -63.2 | -48.2 | 1.3 |
| region:none | moistening L_v∫S (global mean) | -6.59 | -1.54 | 4.3 |
| region:none | work ∫F·u (global mean) | -3.84 | -2.53 | 1.5 |
| region:none | largest |flux| of a column | 6.25e+03 | 2.34e+03 | 2.7 |
| region:none | largest work of a column | 66.7 | 14.2 | 4.7 |
| region:none | clamped fraction | 0 | 0 | 0.0 |
| region:flux250 | heating c_p∫Q (global mean) | -42.8 | -35.9 | 1.2 |
| region:flux250 | moistening L_v∫S (global mean) | 3.2 | 2.68 | 1.2 |
| region:flux250 | work ∫F·u (global mean) | -1.03 | -1.71 | 0.6 |
| region:flux250 | largest |flux| of a column | 3.66e+03 | 2.91e+03 | 1.3 |
| region:flux250 | largest work of a column | 18.7 | 14 | 1.3 |
| region:flux250 | clamped fraction | 0.471 | 0.483 | 1.0 |
| region:nowork | heating c_p∫Q (global mean) | -58.2 | -25.4 | 2.3 |
| region:nowork | moistening L_v∫S (global mean) | -8.45 | -0.191 | 44.3 |
| region:nowork | work ∫F·u (global mean) | -5.27 | -2.27 | 2.3 |
| region:nowork | largest |flux| of a column | 5.57e+03 | 2.16e+03 | 2.6 |
| region:nowork | largest work of a column | 49.4 | 13 | 3.8 |
| region:nowork | clamped fraction | 0 | 0 | 0.0 |

## n1: free runs under the bounds (2020, 10 initialisations, 120 h)

Blow-ups per start kind and setting; the error ratio against the unbounded model over the initialisations finite in both (GraphCast-weighted all-point MSE, normalised units); the energy terms are means over the run (W/m^2).

| start | setting | blow-ups | error ratio vs none (24 / 72 / 120 h) | heating | moistening | work | largest flux | clamped |
|---|---|---|---|---|---|---|---|---|
| none | none | 0/10 | reference | 25.7 | -1.0 | -1.15 | 842 | 0.0% |
| none | flux500 | 0/10 | 1.000 / 0.998 / 0.995 (n=10) | 25.9 | -0.9 | -1.14 | 820 | 2.7% |
| none | flux250 | 0/10 | 1.007 / 0.994 / 0.986 (n=10) | 28.1 | -1.1 | -1.10 | 800 | 16.3% |
| none | nowork | 0/10 | 0.999 / 0.989 / 0.978 (n=10) | 25.5 | -0.8 | -1.16 | 836 | 0.0% |
| none | flux500_nowork | 0/10 | 0.999 / 0.988 / 0.975 (n=10) | 25.7 | -0.8 | -1.15 | 817 | 2.7% |
| region | none | 4/10 | reference | nan | nan | nan | nan | 0.0% |
| region | flux500 | 2/10 | 1.007 / 1.006 / 1.276 (n=6) | nan | nan | nan | nan | 7.9% |
| region | flux250 | 3/10 | 1.020 / 1.015 / 1.237 (n=5) | nan | nan | nan | nan | 30.7% |
| region | nowork | 4/10 | 0.998 / 0.993 / 0.985 (n=6) | nan | nan | nan | nan | 0.0% |
| region | flux500_nowork | 2/10 | 1.004 / 0.999 / 1.278 (n=6) | nan | nan | nan | nan | 8.2% |

The energy terms over the four steps before a blow-up, against the same leads of the runs that stayed finite:

| run | term | before blow-up | finite runs, same leads | ratio |
|---|---|---|---|---|
| region:none | heating c_p∫Q (global mean) | 46.8 | 37.4 | 1.3 |
| region:none | moistening L_v∫S (global mean) | -13.4 | -13.4 | 1.0 |
| region:none | work ∫F·u (global mean) | -1.42 | -1.4 | 1.0 |
| region:none | largest |flux| of a column | 2.06e+03 | 1.79e+03 | 1.2 |
| region:none | largest work of a column | 282 | 9.59 | 29.4 |
| region:none | clamped fraction | 0 | 0 | 0.0 |
| region:flux500 | heating c_p∫Q (global mean) | 12.1 | 15.3 | 0.8 |
| region:flux500 | moistening L_v∫S (global mean) | -0.919 | 3.04 | 0.3 |
| region:flux500 | work ∫F·u (global mean) | -1.61 | -0.918 | 1.8 |
| region:flux500 | largest |flux| of a column | 1.36e+03 | 1.27e+03 | 1.1 |
| region:flux500 | largest work of a column | 17.2 | 9.06 | 1.9 |
| region:flux500 | clamped fraction | 0.045 | 0.04 | 1.1 |
| region:flux250 | heating c_p∫Q (global mean) | 10.6 | 17.8 | 0.6 |
| region:flux250 | moistening L_v∫S (global mean) | -0.191 | 2.4 | 0.1 |
| region:flux250 | work ∫F·u (global mean) | -1.49 | -0.894 | 1.7 |
| region:flux250 | largest |flux| of a column | 2.12e+03 | 1.19e+03 | 1.8 |
| region:flux250 | largest work of a column | 226 | 9.37 | 24.2 |
| region:flux250 | clamped fraction | 0.249 | 0.248 | 1.0 |
| region:nowork | heating c_p∫Q (global mean) | 46.7 | 37.4 | 1.2 |
| region:nowork | moistening L_v∫S (global mean) | -13.2 | -13.2 | 1.0 |
| region:nowork | work ∫F·u (global mean) | -1.6 | -1.41 | 1.1 |
| region:nowork | largest |flux| of a column | 2.02e+03 | 1.77e+03 | 1.1 |
| region:nowork | largest work of a column | 115 | 9.11 | 12.6 |
| region:nowork | clamped fraction | 0 | 0 | 0.0 |
| region:flux500_nowork | heating c_p∫Q (global mean) | 15.3 | 18.1 | 0.8 |
| region:flux500_nowork | moistening L_v∫S (global mean) | -2.36 | 1.99 | 1.2 |
| region:flux500_nowork | work ∫F·u (global mean) | -2 | -0.904 | 2.2 |
| region:flux500_nowork | largest |flux| of a column | 1.71e+03 | 1.24e+03 | 1.4 |
| region:flux500_nowork | largest work of a column | 193 | 8.99 | 21.4 |
| region:flux500_nowork | clamped fraction | 0.0402 | 0.0367 | 1.1 |

## E2: free runs under the bounds (2020, 10 initialisations, 120 h)

Blow-ups per start kind and setting; the error ratio against the unbounded model over the initialisations finite in both (GraphCast-weighted all-point MSE, normalised units); the energy terms are means over the run (W/m^2).

| start | setting | blow-ups | error ratio vs none (24 / 72 / 120 h) | heating | moistening | work | largest flux | clamped |
|---|---|---|---|---|---|---|---|---|
| none | none | 0/10 | reference | 13.8 | 15.1 | -1.66 | 1519 | 0.0% |
| none | flux500_nowork | 0/10 | 0.993 / 0.924 / 0.875 (n=10) | 17.2 | 16.2 | -1.43 | 1413 | 14.6% |
| region | none | 0/10 | reference | 26.8 | 14.7 | -1.53 | 1821 | 0.0% |
| region | flux500_nowork | 0/10 | 1.027 / 0.981 / 0.954 (n=10) | 35.9 | 14.6 | -1.40 | 1879 | 15.4% |

## The stage 10 game on region masks, D-noactor, without and with the bound (31 initialisations, 2020)

| variant | blow-ups without | blow-ups with | 24 h hidden error without | with |
|---|---|---|---|---|
| full | 1 | 1 | 1.136 | 1.175 |
| nudging | 0 | 0 | 1.207 | 1.226 |

## The stage 9 cuts on N1 (2020, 8 initialisations, 72 h), without and with the bound

| cut | blow-ups without | with | first blow-up (h) without | with | 72 h all-MSE gain of the cut (+ = better) without | with |
|---|---|---|---|---|---|---|
| no_moistening | 8 | 8 | 18 | 18 | - | - |
| no_heating | 0 | 0 | - | - | -35.8% | -36.2% |

## The corrector arms against E1 (same initialisations)

Each arm is a checkpoint and the setting it runs under; the ratio is the all-point MSE of the arm over E1 on the initialisations finite in both (< 1 = the arm is better).

| arm | checkpoint : setting | start | blow-ups | error ratio (24 / 72 / 120 h) | n |
|---|---|---|---|---|---|
| E1 | stage10_Dnoactor : flux500_nowork | none | 2/10 | reference |  |
| E1 | stage10_Dnoactor : flux500_nowork | region | 0/10 | reference |  |
| E2 | stage13_E2 : flux500_nowork | none | 0/10 | 0.917 / 0.718 / 0.516 | 8 |
| E2 | stage13_E2 : flux500_nowork | region | 0/10 | 1.013 / 0.888 / 0.700 | 10 |
| E2 unbounded | stage13_E2 : none | none | 0/10 | 0.923 / 0.778 / 0.593 | 8 |
| E2 unbounded | stage13_E2 : none | region | 0/10 | 0.986 / 0.906 / 0.733 | 10 |
| none | stage10_Dnoactor : none | none | 5/10 | 1.021 / 1.174 / 1.584 | 4 |
| none | stage10_Dnoactor : none | region | 2/10 | 0.981 / 1.056 / 1.238 | 8 |

## Decision rules

- **B1** the bound (500 W/m^2 + no work) removes >= 80% of the blow-ups (region starts, cold memory, the moisture cut) and costs N1 <= 3% at 24 h  
  → not supported (removed 5 of 16 = +31.2%; memory none 5→2, memory region 2→0, game region 1→1, no_moistening 8→8; N1 24 h cost -0.1%)
- **B4** the injecting term: a term's mean over the four steps before a blow-up is >= 2x the same leads of the finite runs  
  → supported (none:flux500: work_max x5.2; none:flux250: moist_mean x2.6; none:flux250: work_max x2.3; none:nowork: work_max x4.2; none:flux500_nowork: work_max x4.8; region:none: moist_mean x4.3; region:none: flux_max x2.7; region:none: work_max x4.7; region:nowork: heat_mean x2.3; region:nowork: moist_mean x44.3; region:nowork: work_mean x2.3; region:nowork: flux_max x2.6; region:nowork: work_max x3.8)
- **B2** E2 beats E1 by >= 3% at 72 h with no more blow-ups  
  → supported (blow-ups 0 vs 2; 72 h ratio 0.803 = gain +19.7% on 18 finite pairs)
- **B2** E2 unbounded beats E1 by >= 3% at 72 h with no more blow-ups  
  → supported (blow-ups 0 vs 2; 72 h ratio 0.842 = gain +15.8% on 18 finite pairs)

## Direction

- Blow-ups of the memory model: none 7, flux bound 2, no-work 7, both 2. The energy input is the lever: next, E2 learns where the excess is taken from, and the bound goes into the stage 10.2 training so the column never learns to rely on it.
- On the finite runs the bound gains the memory model +2.1% at 24 h and +14.8% at 72 h (+ = better): the unbounded column was injecting energy the forecast did not need.
