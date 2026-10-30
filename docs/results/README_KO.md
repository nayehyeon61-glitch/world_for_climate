# 실험 결과 (WB2 64×32, CPU 4개)

## 1. 2018년 GraphCast 비교 (`graphcast_2018/`)

- **학습:** 2015–2017년 ERA5, `lie_diffusion` + `chain` 결합, 파라미터 0.74M, CPU로 약 1.1시간.
- **평가:** 2018년, 6일 간격 00 UTC 초기 시각 61개, lead 6–120h, 앙상블 멤버 4개.
- **GraphCast:** WB2 0.25° 예측을 64×32로 면적 보존 평균했습니다.

| 지표 | 값 |
|---|---|
| 72h z500 RMSE, 우리 / GraphCast | 830 / 112 m²/s² (약 7.4배) |
| 72h 전체 변수 중앙값, 우리 ÷ GraphCast | 약 4.3배 |
| persistence보다 나은 변수·lead 쌍 | 6h 40%, 24–72h 88%, 96–120h 61% |
| 기후값보다 나은 변수 | 72h 2%, 120h 0% |
| 앙상블 spread/skill (72h 중앙값) | 0.34 (과신) |

GraphCast는 1979–2017년, TPU 32개, 0.25°로 학습했으므로 예산 차이가 매우 큽니다. 6h에서 지는 것은 인코더→디코더 복원 오차 바닥 때문입니다.

## 2. 새 데이터로 world model 보강 (`rl_refinement_2020/`)

**설정:**
- 위의 2015–2017년 모델을 2018–2019년 새 데이터로 150 step 미세조정했습니다.
- 학습한 것은 DIAMOND denoiser(정책)뿐이고, 나머지는 고정했습니다.
- 2020년 31개 초기 시각에서 test했습니다. 세 모델 모두 같은 초기 시각입니다.

**두 가지 방법:**
- **지도학습 대조군:** stochastic 단계 손실(K=4 rollout)만 씁니다.
- **RL (검증 보상 정책 개선):**
  - 초기 상태마다 M=4 멤버로 K=4 step rollout을 만들고, 보상은 −(GraphCast 가중 MSE)입니다.
  - GRPO식 그룹 정규화 이점으로 가중한 EDM 회귀를 합니다.
  - 같은 지도학습 손실을 anchor로 더합니다.

| 기본 모델 대비 (83개 변수 × 20 lead 평균) | 지도학습 | RL |
|---|---|---|
| 앙상블 평균 RMSE | −0.11% (53% 개선) | **−0.57%** (73% 개선) |
| 멤버 1개 RMSE | +0.42% (34% 개선) | **−1.37%** (90% 개선) |
| CRPS | **−1.22%** | +1.83% |
| spread/skill 중앙값 (기본 0.34) | **0.36** | 0.29 |

**해석:**
- RL은 각 멤버를 정답 쪽으로 당겨 결정론적 정확도를 높였습니다. 효과는 lead가 길수록 커서, 120h 멤버 RMSE는 −1.7%입니다.
- 대신 앙상블 퍼짐이 약 13% 줄어 확률 예측(CRPS)은 나빠졌습니다. 멤버별 MSE 보상은 평균에 가까운 표본을 선호하므로, 이미 과신하던 앙상블이 더 좁아졌습니다.
- 지도학습은 반대로 퍼짐을 조금 넓혀 CRPS를 개선했습니다. 결정론적 정확도는 그대로입니다.
- 두 방법 모두 GraphCast와의 차이에 비하면 효과가 작습니다(2% 미만). 고정된 Lie 구조 전이가 평균 궤적을 정하기 때문입니다.
- 초기 시각이 31개이고 부트스트랩 신뢰구간은 계산하지 않았습니다. 다만 RL의 멤버 개선(90%)과 퍼짐 감소(모든 lead에서 −12~−13%)는 변수와 lead 전반에서 일관됩니다.

**다음 단계 후보:**
- 보상을 그룹 단위의 적정 점수(fair CRPS)로 바꿉니다. 멤버 m의 보상을 −(|x_m − y| − Σ_j |x_m − x_j| / (2(M−1)))로 두면 정확도와 다양성을 함께 보상합니다.
- RL 단계에서 구조 전이(Lie 생성기)도 함께 학습합니다.

## 재현

```bash
python scripts/refine.py --checkpoint runs/main/checkpoint.pt --config configs/refine_wb2.yaml \
    --data-dir DATA2020 --mode rl --out runs/refine_rl            # --mode supervised 는 대조군
python scripts/compare_graphcast.py --checkpoint runs/refine_rl/checkpoint.pt --data-dir DATA2020 \
    --init-stride 6 --out runs/test2020/rl.json
python scripts/report_refinement.py base=runs/test2020/base.json \
    supervised=runs/test2020/supervised.json rl=runs/test2020/rl.json --out-dir runs/test2020/report
```
