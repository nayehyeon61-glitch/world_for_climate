## Self-explainability (6 h pairs)

| arm | Lie codes | physics codes | loss: persistence | MDL-selected | full K | floor | explained fraction | mean k* | codes used / perplexity | physics share |
|---|---|---|---|---|---|---|---|---|---|---|
| C (wb2_64x32_theory_vq_lie) | 16 | 0 | 0.7642 | 0.6574 | 0.6623 | 0.5302 | 0.452 | 2.16 | 16 / 5.7 | – |
| A' (wb2_64x32_theory_continuous) | cont. | 0 | 0.7642 | 0.6497 | 0.6493 | 0.5302 | 0.485 | 2.69 | continuous | – |

## Analog transfer (program of pair i executed on its nearest state j ≥ 7 days away)

| arm | persistence | delta copy | random program | **transferred** | oracle | transfer gain | delta gain | transfer beats delta |
|---|---|---|---|---|---|---|---|---|
| C | 0.7349 | 1.0285 | 0.6606 | **0.6534** | 0.6380 | 0.840 | -3.198 | 100% |
| A' | 0.7349 | 1.0285 | 0.6520 | **0.6446** | 0.6303 | 0.861 | -2.959 | 100% |

## Length generalisation (12 h pairs, trained on 6 h)

| arm | max steps | persistence | selected | full | floor | mean k* | explained fraction |
|---|---|---|---|---|---|---|---|
| C | 3 | 1.3984 | 1.0044 | 1.0047 | 0.5533 | 2.73 | 0.463 |
| C | 6 | 1.3984 | 0.9760 | 0.9862 | 0.5533 | 4.26 | 0.496 |
| A' | 3 | 1.3984 | 1.0131 | 1.0128 | 0.5533 | 2.95 | 0.452 |
| A' | 6 | 1.3984 | 0.9534 | 0.9539 | 0.5533 | 5.15 | 0.523 |
