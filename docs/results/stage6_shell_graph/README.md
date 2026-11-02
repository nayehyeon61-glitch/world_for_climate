# 6단계 — 쉘 그래프 결합

가설·구현·팔·예측(한국어): [`../README_KO.md`](../README_KO.md) 6단계 절.

| 팔 | 설정 | 상태 |
|---|---|---|
| H0 | `configs/wb2_pyramid_nodrop.yaml`: 사슬 혼합, 독립 쉘 (skip dropout을 끈 4단계 P) | 학습 완료 후 원인 분석 |
| H1 | `configs/wb2_h1.yaml`: H0 + 쉘 그래프 혼합기 | 학습 대기 |
| H2 | `configs/wb2_h2.yaml`: H1 + 생성원 그래프 결합 | 학습 대기 |

결과 파일(학습 후 추가): `causes.md`·`causes.png`(`scripts/report_causes.py`: 오차 분해, knock-out, 학습된 쉘 그래프), `causes_H*.json`.
