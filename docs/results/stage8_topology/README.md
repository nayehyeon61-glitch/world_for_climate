# 8단계 — 동역학적 대기의 위상 (전이 비용)

해석(한국어): [`../README_KO.md`](../README_KO.md) 8단계 절. 설계 문서: [8단계 설계: 동역학적 위상(전이 비용) 학습](https://claude.ai/artifact/Y8G5FeKMbF77Vc8voYJ9B5).

물리적 전이 비용으로 정의한 근방이 대기 상태공간에 새 위상을 만드는지, 기하만 바꾸는지, 그리고 그 근방이 예측에 도움이 되는지 시험합니다. 1차는 고정한 세계 모델(P−nodrop)의 latent 위에 커뮤트 타임 임베딩 머리(CTWM, arXiv 2610.01373)만 2015–17로 학습하고 2018에서 T1–T3를 잽니다.

결과가 나오면 이 폴더에 들어갑니다.
- `topology.md`, `topology.png`: `scripts/report_topology.py` 출력 — T1(커뮤트 타임 Spearman, 시간 간격별 거리), T2(분산 스펙트럼 기울기, H0 지속성, 레짐 체류 시간, 지역 패치), T3(유사 사례 예보, 레짐별, 고정 모델 오차와의 상관).
- `topology.json`: `scripts/eval_topology.py analyse` 출력.
