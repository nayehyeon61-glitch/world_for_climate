# 4단계 — 그래프 V-Net so(3) 피라미드

설계 문서(한국어): https://claude.ai/artifact/HyDfQMgse2dCX9CHhku21j
전체 단계 개요와 가설·구현 요약: [`../README_KO.md`](../README_KO.md) 4단계 절.

## 팔

| 팔 | 설정 | 상태 |
|---|---|---|
| S | 1단계 단일 스케일, independent 결합 (`stage1_world_model/ablation_2018/`의 체크포인트) | 완료 |
| P | `configs/wb2_pyramid.yaml`: 레벨 1·2·3 상태, so(3) 피라미드 + skip | 학습 대기 |
| P−skip | `configs/wb2_pyramid_noskip.yaml`: 레벨 1 상태만 | 학습 대기 |

## 결과 파일 (학습 후 추가)

- `pyramid.md`, `pyramid.png`: `scripts/report_pyramid.py` 출력 — S·P·P−skip 스코어카드(2018, 61개 초기 시각), 72–120h 판정, 앙상블 CRPS·spread/skill, 레벨별 에너지, 레벨별 전이, 예측 1–3 판정.
- `pyramid_eval_P.json`, `pyramid_eval_noskip.json`: `scripts/eval_pyramid.py` 출력.
- `compare_graphcast_P.json`, `compare_graphcast_noskip.json`: 1단계 프로토콜 원자료.
