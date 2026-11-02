# 9단계 — 물리 중심 세계 모델

해석(한국어): [`../README_KO.md`](../README_KO.md) 9단계 절. 설계 문서: [9단계 설계: 물리 중심 세계 모델](https://claude.ai/artifact/CsnpM54EbtjvzPZXTdPpzt).

상태는 물리 변수 그대로이고, 미분 가능한 원시방정식 스펙트럴 코어(T31) 위에 신경망 칼럼 물리를 다중 스텝 롤아웃으로 학습합니다. 고정한 모델 위에서는 Ha & Schmidhuber 2018 방식의 제어기(N3)가 마스킹 예측 게임을 합니다.

- `n0_2018/`: N0(학습 없는 코어, 경계층 마찰 포함)의 2018년 31개 초기 시각 GraphCast 비교 — `scorecard.md`, `rmse_vs_lead.png`, `acc_vs_lead.png`, `compare_graphcast_summary.json` (`scripts/compare_graphcast.py --init-stride 6 --lead-steps 20`, `scripts/report_comparison.py`).
- `n0_nodrag_2018/`: 같은 비교, 경계층 마찰 없음(원인 분석; 3개 초기 시각이 48–66h에 발산해 72h 이후 평균은 NaN).
- `n0_knockouts.json`: N0 항별 knock-out(8개 초기 시각, 6·24·72h RMSE).
