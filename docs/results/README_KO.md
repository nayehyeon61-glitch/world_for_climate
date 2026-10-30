# 실험 결과 (WB2 64×32, CPU 4개)

가설이 바뀔 때마다 단계(Stage)를 나눕니다. 한 단계 안의 세부 수정(보상 변경, 절제 실험, 버그 수정)은 같은 단계에 둡니다.

| 단계 | 가설 | 상태 | 결과 폴더 |
|---|---|---|---|
| 1 | Lie 군 Shell-GNN + DIAMOND 잠재 세계 모델. 쉘을 "액체 지네 로봇"처럼 사슬로 결합 | 완료: 2018 GraphCast 비교, 결합 절제 실험 | `stage1_world_model/` |
| 2 | 학습된 세계 모델을 새 연도 데이터로 RL(검증 보상 정책 개선) 보강 | 완료: 2020 held-out 시험, 지도학습 대조군 | `stage2_rl_refinement/` |
| 3 | World Theory Model: 6시간 변화를 so(3) 원시 연산의 짧은 프로그램으로 설명 (Learning-to-Theorize) | 진행 중: 1차(VQ-Lie vs 연속 대조) 학습 | `stage3_theory/` |

---

## 1단계 — Lie-GNN + DIAMOND 세계 모델

### 1.1 2018년 GraphCast 비교 (`stage1_world_model/graphcast_2018/`)

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

### 1.2 쉘 결합 절제 실험: chain vs independent (`stage1_world_model/ablation_2018/`)

같은 설정(2100 스텝)으로 `shell_coupling: independent`를 학습하고, 1.1과 같은 61개 초기 시각에서 평가했습니다. 변화율은 (independent − chain) / chain입니다.

| 83개 변수 × 20 lead 평균 | independent |
|---|---|
| 앙상블 평균 RMSE | **−0.74%** (변수·lead 쌍의 58% 개선) |
| 멤버 1개 RMSE | −0.64% (56% 개선) |
| CRPS | −1.17% |
| spread/skill 중앙값 (chain 0.34) | 0.35 |

- 대표 변수: z500 RMSE 6h −9.4%, 24h −3.0%, 72h −0.8%, 120h −4.6%; T850 120h −10.6%. 반면 T2m +3.9%(6h) ~ +10.1%(120h), U850 24–72h 약 +2%로 지표 변수는 chain이 다소 낫습니다.
- **판정:** 이 규모에서는 쉘 간 사슬 결합이 이득을 주지 못했고 독립형이 근소하게 낫습니다. 차이는 작고 변수별로 엇갈려 61개 초기 시각의 변동 범위 안으로 봐야 합니다. 지네형 가설은 대규모(0.25°, GPU) 학습에서 다시 판정합니다.

---

## 2단계 — 새 데이터로 world model 보강 (`stage2_rl_refinement/`)

**설정:**
- 1단계의 2015–2017년 모델을 2018–2019년 새 데이터로 150 step 미세조정했습니다.
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

**이 단계의 후속 후보 (세부 수정):**
- 보상을 그룹 단위의 적정 점수(fair CRPS)로 바꿉니다. 멤버 m의 보상을 −(|x_m − y| − Σ_j |x_m − x_j| / (2(M−1)))로 두면 정확도와 다양성을 함께 보상합니다.
- RL 단계에서 구조 전이(Lie 생성기)도 함께 학습합니다.

---

## 3단계 — World Theory Model (`stage3_theory/`)

**가설:** 관측 쌍 (Z_t, Z_{t+6h})의 변화를 예측하는 대신, 공유 어휘(codebook)의 원시 연산을 조합한 가장 짧은 프로그램으로 **설명**한다(Baek et al. 2026, "Learning to Theorize the World from Observation"). 원시 연산은 so(3) 원소이므로 실행기는 고정된 군 작용이고 합성은 닫혀 있습니다. 설계 문서(한국어): https://claude.ai/artifact/UW5M3kDK6rgze7xjT3ivPV

**구현 (`models/theory.py`, `eval/transfer.py`):**
- 전이 `lie_program`: 목표 조건부 프로그래머가 (쉘, 패치)마다 한 원시 연산을 제안 → EMA 벡터 양자화(코드 0 = 항등) → 군 작용 실행 → 패치별 MDL로 길이 k* 선택. `codebook_size: 0`이면 연속 프로그램(대조군 Cont-Mono).
- `theory` 학습 단계: 인코더·디코더는 1단계 체크포인트로 고정, 프로그래머와 어휘만 학습. 모든 접두 프로그램을 목표로 당기고 MDL은 선택만 합니다.
- 평가 `scripts/eval_transfer.py`: 자기 설명력(persistence·복원 바닥 대비), 유사 상태 전이(delta copy·무작위 프로그램·oracle 대비), 12시간 쌍의 길이 일반화.

### 3.1 1차 실험: C = VQ-Lie(어휘 16, K=3) vs A' = 연속 프로그램 (`stage3_theory/phase1_vq_lie_vs_continuous.md`)

2018년 6h 쌍 183개(4일 간격), 같은 고정 인코더·디코더(1단계 체크포인트), 각 1500스텝(CPU 약 55분).

| 지표 | C (VQ-Lie) | A' (연속) |
|---|---|---|
| 설명 가능한 변화 중 설명한 비율 (persistence 0.764 → 복원 바닥 0.530) | 45.2% | **48.5%** |
| 평균 프로그램 길이 k* | 2.16 | 2.69 |
| 유사 상태 전이 손실 (oracle 대비 이득) | 0.653 (0.84) | **0.645 (0.86)** |
| 무작위 프로그램 손실 (대조군) | 0.661 | 0.652 |
| 12h 쌍, K=6 설명 손실 (persistence 1.398, 바닥 0.553) | 0.976 | **0.953** |

- **판정:** 이산 Lie 어휘가 전이·일반화를 개선한다는 가설 1은 이 규모에서 지지되지 않습니다. 연속 프로그램이 모든 지표에서 2~3%p 낫습니다. 이산화의 이득은 압축뿐입니다((쉘, 패치, 스텝)당 4비트로 전이 성능의 97% 유지).
- 두 모델 모두 **무작위 프로그램이 전이 프로그램과 거의 같습니다**. 어휘가 잡은 것은 전역 공통 운동(평균류 이송)이고, 상태 특이적 기제는 persistence→oracle 격차의 약 7%입니다. 2차의 질문은 물리 코드가 이 generic 격차를 메우는지입니다.
- 6h로만 학습한 프로그램을 12h 쌍에 6스텝까지 합성하면 설명이 더 좋아져(길이 OOD) 합성 구조 자체는 유효합니다.

### 3.2 2차 실험: Lie 위의 잔차 physics 코드 (진행 중)

V = V_Lie + V_phys + V_noise 분해(수치예보의 역학 코어 + 모수화 분할)를 따라, Lie 스텝 뒤에 칼럼 국소 physics 코드(복사·수직 혼합(플럭스 형태, 보존)·지표 교환)를 실행합니다(`models/physics_codes.py`). E = hybrid(Lie 16 + physics 15), D = physics-only(Lie 어휘는 항등만). 평가에는 개입 검사(TISR=0 반사실, 가족 제거)와 2020년 불변성 검사가 들어갑니다. 결과는 `stage3_theory/`에 추가합니다. 3차(예측용 사전 정책)는 아직입니다.

---

## 재현

```bash
# 1단계
python scripts/train.py --config configs/wb2_64x32.yaml --data-dir DATA              # chain
python scripts/train.py --config configs/wb2_64x32.yaml --data-dir DATA \
    --set model.lie.shell_coupling=independent output_dir=runs/independent          # 절제
python scripts/compare_graphcast.py --checkpoint runs/wb2_64x32/checkpoint.pt --data-dir DATA --init-stride 3
python scripts/report_refinement.py chain=runs/wb2_64x32/compare_graphcast.json \
    independent=runs/independent/compare_graphcast.json --out-dir runs/independent/report
# 2단계
python scripts/refine.py --checkpoint runs/main/checkpoint.pt --config configs/refine_wb2.yaml \
    --data-dir DATA2020 --mode rl --out runs/refine_rl            # --mode supervised 는 대조군
python scripts/compare_graphcast.py --checkpoint runs/refine_rl/checkpoint.pt --data-dir DATA2020 \
    --init-stride 6 --out runs/test2020/rl.json
python scripts/report_refinement.py base=runs/test2020/base.json \
    supervised=runs/test2020/supervised.json rl=runs/test2020/rl.json --out-dir runs/test2020/report
# 3단계
python scripts/train.py --config configs/wb2_theory_lie.yaml --data-dir DATA --init-from runs/wb2_64x32/checkpoint.pt
python scripts/eval_transfer.py --checkpoint runs/wb2_theory_lie/checkpoint.pt --data-dir DATA --stride 4
```
