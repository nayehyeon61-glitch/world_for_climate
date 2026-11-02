# 7단계 — 예측 게임과 GNN feature 제어

해석(한국어): [`../README_KO.md`](../README_KO.md) 7단계 절. 설계 문서: [7단계 설계: 예측 게임과 GNN feature 제어](https://claude.ai/artifact/H8az4WzNqJvUuvWYYGz1LU).

고정한 세계 모델(P−nodrop) 위에서 GNN feature를 조절하는 작은 제어기만 마스킹 예측 게임의 보상으로 학습하고(2018–19), 2020년에 시험합니다. 게임은 36h(6스텝) 시퀀스이고, 스텝마다 확률 0.5로 가려진 실제 관측을 다시 보여 주며, 각 스텝의 행동은 reward-to-go로 평가합니다.

결과가 나오면 이 폴더에 들어갑니다.
- `game.md`, `game.png`: `scripts/report_game.py` 출력 — 마스크 종류별 2020 게임 점수(제어기, 섞인 위치 행동, 진단량 없는 제어기; 고정 모델 대비 쌍별 90% 구간), 마스킹 없는 6–72h 예보 변화, 학습 곡선.
- `game_eval_*.json`: `scripts/eval_game.py` 출력. `causes_2020_*.json`: `scripts/eval_causes.py --controller --deterministic` 출력.
