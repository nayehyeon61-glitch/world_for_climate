# 5단계 원인 분석 — 오차 분해와 평가 시 ablation

해석(한국어): [`../../README_KO.md`](../../README_KO.md) 5.1절.

- `causes.md`, `causes.png`: `scripts/report_causes.py` 출력 — S·P·P−skip·P−nodrop(skip dropout 0)·G1의 오차 분해(자동부호화기 바닥 rec + 잠재 동역학 dyn), 변수 평균 바닥 비율, 평가 시 ablation(denoiser 끔, G1의 복합 군·수직 이동·혼합 끔)의 쌍별 90% bootstrap 구간.
- `causes_*.json`: `scripts/eval_causes.py` 출력(2018 검증 기간에서 고르게 뽑은 31개 초기 시각, 72h, 단일 멤버).

재현:

```
python scripts/eval_causes.py --checkpoint runs/wb2_g1/checkpoint.pt --data-dir DATA --variants no_denoiser composite shift mixing
python scripts/eval_causes.py --checkpoint runs/wb2_pyramid/checkpoint.pt --data-dir DATA --variants no_denoiser
python scripts/eval_causes.py --checkpoint runs/wb2_pyramid_nodrop/checkpoint.pt --data-dir DATA --variants no_denoiser
python scripts/report_causes.py S=runs/independent/causes.json P=runs/wb2_pyramid/causes.json \
    "P-skip=runs/wb2_pyramid_noskip/causes.json" "P-nodrop=runs/wb2_pyramid_nodrop/causes.json" G1=runs/wb2_g1/causes.json --out-dir docs/results/stage5_causal_lie/causes
```
