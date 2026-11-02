# 4단계 — 그래프 V-Net so(3) 피라미드

설계 문서(한국어): https://claude.ai/artifact/HyDfQMgse2dCX9CHhku21j
가설·구현 요약과 결과 해석(한국어): [`../README_KO.md`](../README_KO.md) 4단계 절.

## 팔

| 팔 | 설정 | 상태 |
|---|---|---|
| S | 1단계 단일 스케일, independent 결합 (`stage1_world_model/ablation_2018/`의 체크포인트) | 완료 |
| P | `configs/wb2_pyramid.yaml`: 레벨 1·2·3 상태, so(3) 피라미드 + skip | 완료 (학습 ~1h50, 비교 ~1h50) |
| P−skip | `configs/wb2_pyramid_noskip.yaml`: 레벨 1 상태만 | 완료 (학습 ~1h20, 비교 ~17분) |

## 결과 파일

- `pyramid.md`, `pyramid.png`: `scripts/report_pyramid.py` 출력 — S·P·P−skip 스코어카드(2018, 61개 초기 시각), 72–120h 판정, 앙상블 CRPS·spread/skill, 레벨별 에너지, 레벨별 전이, 예측 1–3 판정.
- `pyramid_eval_P.json`, `pyramid_eval_noskip.json`: `scripts/eval_pyramid.py` 출력 (183개 6h 쌍, 유사 상태 간격 ≥ 7일).
- `compare_graphcast_summary_P.json`, `compare_graphcast_summary_noskip.json`: 1단계 프로토콜 요약(전체 원자료는 용량 때문에 저장소에 넣지 않음).

## 판정 요약

- 예측 1 (거친 레벨 전이): 미지지. 상태 특이적 정보는 미세 레벨 δ⁽²⁾에 있음(일치 − 무작위 +0.126, 거친 레벨 +0.003).
- 예측 2 (에너지 스펙트럼): 부분 지지. 위도 구조는 맞지만 P의 에너지는 L2가 95%.
- 예측 3 (72–120h z500): P 미지지(+7.3%), P−skip 형식상 지지(−9.3%). 다만 세 팔 모두 이 구간에서 기후값보다 RMSE가 커서, 차이는 오차 포화 속도의 차이로 해석.
