# 실험 결과 (WB2 64×32, CPU 4개)

가설이 바뀔 때마다 단계(Stage)를 나눕니다. 한 단계 안의 세부 수정(보상 변경, 절제 실험, 버그 수정)은 같은 단계에 둡니다.

| 단계 | 가설 | 상태 | 결과 폴더 |
|---|---|---|---|
| 1 | Lie 군 Shell-GNN + DIAMOND 잠재 세계 모델. 쉘을 "액체 지네 로봇"처럼 사슬로 결합 | 완료: 2018 GraphCast 비교, 결합 절제 실험 | `stage1_world_model/` |
| 2 | 학습된 세계 모델을 새 연도 데이터로 RL(검증 보상 정책 개선) 보강 | 완료: 2020 held-out 시험, 지도학습 대조군 | `stage2_rl_refinement/` |
| 3 | World Theory Model: 6시간 변화를 so(3) 원시 연산의 짧은 프로그램으로 설명 (Learning-to-Theorize) | 완료: 1차(VQ-Lie vs 연속), 2차(잔차 physics 코드 미지지, 2020 불변성 확인) | `stage3_theory/` |
| 4 | 그래프 V-Net so(3) 피라미드: 정이십면체 계층으로 Lie 대수를 다운스케일하고 세계 모델이 거친 레벨에서 학습 | 완료: P·P−skip 2018 GraphCast 비교, 예측 1–3 판정(1 미지지, 2 부분, 3 P 미지지·P−skip 형식상 지지) | `stage4_pyramid/` |
| 5 | 현실 정합 인과 Lie 세계 모델: 복합 군 (SO(3) × Vert) ⋉ Fib(이송 + 상승·하강 + 물리 가족), 물리 진단량을 GNN 입력으로, critic이 현실 정합성 판정 | G1 완료(P1 미지지), 원인 분석(자동부호화기 바닥 = skip dropout, P−nodrop으로 확인; denoiser 잡음), G2 학습 중, G3 보류 | `stage5_causal_lie/` |
| 6 | 쉘 그래프 결합: 쉘 사이의 사슬(s±1) 연결을 학습된 쉘 그래프로 바꾸고, 같은 패치의 쉘 생성원도 그래프로 결합 | 구현 완료, 7단계 뒤 H1만 학습(H2 = Lie 생성원 결합은 보류, 10/31 결정), H0 = P−nodrop | `stage6_shell_graph/` |
| 7 | 예측 게임과 GNN feature 제어: 학습된 물리 예측 모델을 고정하고, 마스킹된 데이터를 예측하는 게임에서 GNN feature를 조절하는 작은 제어기만 보상으로 학습한 뒤 전체를 시험 (Ha의 World Models) | 구현 완료, C1·C1-grad·C1-open 학습 대기(G2 뒤) | `stage7_prediction_game/` |

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

### 3.2 2차 실험: Lie 위의 잔차 physics 코드 (`stage3_theory/phase2_physics_codes.md`)

V = V_Lie + V_phys + V_noise 분해(수치예보의 역학 코어 + 모수화 분할)를 따라, Lie 스텝 뒤에 칼럼 국소 physics 코드(복사·수직 혼합(플럭스 형태, 보존)·지표 교환)를 실행합니다(`models/physics_codes.py`). 같은 2018년 6h 쌍 183개, 같은 고정 인코더·디코더, 각 1500스텝.

- E1 = hybrid(Lie 16 + physics 15), 게이트 없음: 양자화기가 항등 코드를 고르지 않아 모든 패치에 매 스텝 physics가 들어가 Lie 프로그램을 방해했습니다.
- E2 = E1 + (쉘, 패치)별 학습 게이트(시그모이드, 닫힌 상태로 시작해 C와 같은 모델에서 출발).
- D = physics-only(Lie 어휘는 항등만).

| 지표 | C (VQ-Lie) | E1 | **E2** | D |
|---|---|---|---|---|
| 설명한 비율 (persistence 0.764 → 바닥 0.530) | **45.2%** | 34.0% | 44.7% | 14.1% |
| 평균 프로그램 길이 k* | 2.16 | 0.85 | 1.51 | 0.87 |
| physics 사용 스텝 / 오차 감소 기여 | – | 100% / 0% | 19% / −46% | 34% / 100% |
| physics 사용 쉘 (지표 → 상층) | – | 전 쉘 | 경계층 95% | 지표 93%, 하부 대류권 62% |
| 전이 손실 (무작위 / oracle) | **0.653** (0.661 / 0.638) | 0.690 (0.697 / 0.676) | 0.655 (0.659 / 0.642) | 0.717 (0.719 / 0.714) |
| 상태 특이적 몫 = (무작위 − 전이) / (persistence − oracle) | **7.4%** | 12% | 4.7% | 11% |
| 12h 쌍 K=6 설명 비율 | **49.6%** | 41.6% | 48.7% | 11.6% |

개입 검사(설계 문서의 인과 검사):

| 검사 | E2 | D |
|---|---|---|
| TISR := 밤 반사실, 잠재 효과: 복사 코드가 쓰인 패치 / 코드 없는 패치 | 3.7e-3 / 9.0e-5 (**41배**) | 3.3e-5 / 1.0e-5 |
| 위 반사실의 복원 2 m 기온 변화 | −0.002 (정규화 단위) | −0.0001 |
| 가족 제거 시 손실 증가: 복사 / 혼합 / 지표 | +0.16% / +0.47% / +0.09% | +0.29% / **+2.84%** / +0.01% |

2020년 불변성(2015–17로 배운 어휘를 2020년 쌍 183개에 적용):

| 팔 | Lie 코드 사용 분포 TV 거리 (2018 vs 2020) | physics 쉘 분포 TV | k* | 설명 비율 | 전이 이득 |
|---|---|---|---|---|---|
| C | 0.012 | – | 2.16 → 2.13 | 45.2% → 45.7% | 0.840 → 0.845 |
| E2 | 0.011 | 0.000 | 1.51 → 1.46 | 44.7% → 45.2% | 0.862 → 0.861 |

- **판정:** 잔차 physics 코드는 설명력도 전이도 개선하지 않습니다(가설 2 미지지). 게이트를 단 E2도 C와 같은 수준이고, physics가 켜진 스텝에서는 오히려 오차를 늘립니다(기여 −46%). physics만으로는 설명 가능한 변화의 14%만 설명하므로, 6시간 변화의 대부분은 Lie 이송이 담당합니다.
- **인과 구조는 맞지만 크기가 작습니다.** E2에서 일사를 밤으로 바꾸면 복사 코드가 쓰인 곳에서만 41배 큰 반응이 나와 "복사 코드 = 일사 반응"이라는 대응은 학습되었습니다. 그러나 복원된 기온 변화는 무시할 수준입니다. D에서는 이런 국소화가 없고, 측정 가능한 인과 역할을 가진 가족은 수직 혼합뿐입니다(제거 시 +2.84%).
- **어휘는 해를 넘어 안정적입니다.** 2020년에도 코드 사용 분포가 거의 같고(TV 0.01) 설명 비율·전이 이득이 그대로입니다. 다만 1차에서 본 것처럼 전이 프로그램과 무작위 프로그램의 차이가 작아(상태 특이적 몫 5–7%), 안정적인 것은 주로 전역 공통 운동입니다.
- **5단계로 넘기는 이유:** physics를 Lie 군 밖의 잔차로 붙이면 Lie가 이미 설명한 것과 경쟁할 뿐입니다. 5단계는 physics를 군 안에 넣어(복합 군 (SO(3) × Vert) ⋉ Fib, 이송된 좌표계에서 물리 적용), 진단량을 GNN 입력으로 주고, critic으로 현실 정합성을 판정합니다.

---

## 4단계 — 그래프 V-Net so(3) 피라미드 (`stage4_pyramid/`)

**가설:** 정이십면체 계층(레벨 1/2/3 = 42/162/642 노드)을 그래프 V-Net처럼 수축·확장하면서 Lie 대수를 레벨별로 다운스케일한 so(3) 피라미드(ω⁽⁰⁾ 12패치, δ⁽¹⁾ 42, δ⁽²⁾ 162) 위에서 세계 모델을 학습하면, 전역 공통 운동(거친 레벨)과 상태 특이 기제(미세 레벨)가 구조적으로 분리되어 전이가 나아지고 72–120h 롤아웃이 안정된다. 3단계 1차의 "무작위 프로그램 ≈ 전이 프로그램"이 동기입니다. 설계 문서(한국어): https://claude.ai/artifact/HyDfQMgse2dCX9CHhku21j

**구현 (`models/pyramid_graphs.py`, `vnet.py`, `pyramid_transition.py`, `eval/pyramid.py`):**
- 잠재 상태는 레벨 1·2·3 노드를 쌓은 846개 노드에 있습니다(P−skip은 레벨 1의 42개만). 전이·잔차·DIAMOND denoiser는 쌓인 노드 집합(레벨별 다중메시 + 부모–자식 엣지) 위에서 그대로 돌아갑니다.
- 인코더(수축)는 GraphCast 규칙을 레벨마다 적용합니다: 레벨 ℓ 노드는 풀링된 ℓ+1 상태에 더해 반경 0.6 × (레벨 ℓ 변 길이)의 격자점을 직접 합칩니다. 디코더(확장)는 가장 거친 상태에서 포함 삼각형으로 올라오며 레벨 상태를 skip으로 받고, 학습 중 확률 0.5로 skip을 끊습니다.
- so(3) 피라미드 전이: 쌓인 그래프 위의 Shell-GNN 하나가 속도장을 내고, 계층적 Killing 사영(부모 원소를 사전값으로, 작은 패치가 못 보는 축 방향 성분 보완)으로 ω⁽⁰⁾, δ⁽¹⁾, δ⁽²⁾를 얻어 스케일 사슬 g = exp(ω⁽⁰⁾)exp(δ⁽¹⁾)exp(δ⁽²⁾)로 모든 레벨을 이송합니다. 세부 회전은 joint 벌칙, latent 손실은 레벨 가중(1.0/0.5/0.25).
- 평가 `scripts/eval_pyramid.py` + `report_pyramid.py`: 레벨별 회전 에너지(위도대·쉘별), 레벨별 유사 상태 전이(거친/미세 레벨을 섞어 실행, 무작위 쌍 대조), 1단계 GraphCast 프로토콜과 결합한 S·P·P−skip 스코어카드.

**검증할 예측:** (1) 거친 레벨 회전의 전이 이득은 무작위 대비 크고(격차 ≥ 0.2) 미세 레벨은 작다(< 0.1); (2) 레벨별 에너지는 L0 > L1 > L2, L2는 중위도 > 열대; (3) P는 S(1단계 independent)보다 72–120h z500 RMSE가 낮고 ACC가 높다.

### 4.1 결과: S vs P vs P−skip (`stage4_pyramid/pyramid.md`, 2018년 61개 초기 시각, 1단계 프로토콜 그대로)

| 지표 | S (단일 스케일) | P (피라미드 + skip) | P−skip (레벨 1만) |
|---|---|---|---|
| z500 RMSE 6h / 72h / 120h (m²/s²) | **260** / 823 / 1032 | 353 / 862 / 1145 | 435 / **811** / **876** |
| z500 72–120h 평균 RMSE (S 대비) / ACC 변화 | 933 | +7.3% / −0.014 | **−9.3% / +0.015** |
| t2m RMSE 72h / 120h (K) | 4.21 / 5.45 | 3.90 / 4.80 | **3.89 / 4.23** |
| mslp RMSE 72h / 120h (hPa) | **7.46 / 8.36** | 7.80 / 9.91 | 7.87 / 8.99 |
| z500 spread/skill 72h (1이 이상적) | 0.255 | **0.320** | 0.255 |
| 학습 끝 복원 손실 (dynamics 단계 rec) | **0.26** | 0.53 | 0.73 |
| 레벨별 회전 에너지 비율 L0 / L1 / L2 | – | 3% / 2% / **95%** | 57% / 43% |
| 전이 이득 일치 − 무작위: 거친 / 미세 | – | +0.003 / **+0.126** | −0.10 / −0.35 (자기 회전 설명 1.7%, 잡음 수준) |

- **예측 1 (거친 레벨이 전이, 미세 레벨은 아님): 미지지, 오히려 반대.** P에서 12·42 패치 레벨의 회전은 전체 에너지의 5%뿐이라 바꿔 넣어도 결과가 그대로입니다(0.992 vs 무작위 0.988). 상태 특이적 정보는 162 패치 레벨 δ⁽²⁾에 있습니다(일치 0.660 vs 무작위 0.534). 물리적으로도 말이 됩니다: 미세 레벨 rms 2.16°/6h는 10 m/s 이송(약 1.9°/6h)과 같은 크기이고, 행성 규모 패치 위의 Killing 사영은 파동의 회전이 서로 상쇄되어 거의 0입니다. 즉 "전역 공통 운동"은 거의 강체 회전이 없고, 이송은 종관 규모에서 일어납니다.
- **예측 2 (에너지 L0 > L1 > L2, L2는 중위도 > 열대): 부분 지지.** 위도 구조는 맞습니다(P의 L2 중위도/열대 3.26, P−skip 2.10). 그러나 P의 에너지는 미세 레벨로 갈수록 커집니다. P−skip의 단조 감소는 레벨이 둘뿐인 결과입니다.
- **예측 3 (72–120h z500 개선): P는 미지지(+7.3%), P−skip은 형식상 지지(−9.3%)지만 해석에 주의가 필요합니다.** 세 팔 모두 72h 이후 z500 RMSE가 기후값(789)보다 큽니다. 이 구간의 RMSE 차이는 예보 기술이 아니라 오차가 포화되는 속도의 차이입니다. P−skip은 42 노드 상태라 복원이 가장 나쁘고(6h z500 +67%) 예측이 매끄러워 오차가 천천히 커집니다. 72h ACC는 오히려 S보다 낮고(0.295 vs 0.335) 평균 ACC 이득은 96–120h에서만 나옵니다. 자기 회전이 설명하는 몫이 1.7%뿐이라 so(3) 이송도 거의 쓰이지 않습니다.
- **P의 짧은 리드 손해는 V-Net 자동부호화기 때문입니다.** dynamics 단계 끝 복원 손실이 S의 두 배(0.53 vs 0.26)이고, 6h RMSE가 변수에 따라 5–40% 나쁩니다(z500 +35%, q700 +40%). 대신 P는 t2m(72h −7.4%, 120h −11.8%)과 z500 spread/skill(72h 0.32 vs 0.26)이 낫습니다.
- **5단계에 주는 의미:** 5단계 G 팔은 P 위에 복합 군을 얹으므로 G0 = P와 비교합니다(같은 자동부호화기). 미세 레벨 δ⁽²⁾가 이송을 담당한다는 결과는 5단계의 Vert·Fib 생성자를 같은 Cartan 좌표에서 읽는 근거가 됩니다. 자동부호화기 용량(레벨 2·3 skip의 복원)을 늘리는 것은 별도 후속 과제로 남깁니다.

---

## 5단계 — 현실 정합 인과 Lie 세계 모델 (`stage5_causal_lie/`)

**가설:** 대기 방정식 dX/dt = F(X)의 각 항은 벡터장, 곧 Lie 대수의 생성자이고, exp(Δt F)는 군 원소, 순차적 과정은 군의 곱입니다(수치예보의 과정 분할 = Lie–Trotter/Strang). 3단계 2차에서 physics를 Lie 군 밖의 잔차로 붙였을 때 Lie와 경쟁만 했으므로, physics를 **군 안에** 넣습니다: 칼럼 잠재 상태 위의 복합 군 G = (SO(3) × Vert) ⋉ Fib, 즉 수평 이송(SO(3), Killing 사영) × 수직 이동(Vert = 쉘 축 평행이동, 상승·하강) ⋉ 물리 가족(Fib = 아핀 생성자: 복사 편향, 쉘 라플라시안 모드 위의 교환 가능한 수직 혼합, 보존형 지표 교환, 게이트 응결). 반직접곱은 "이송된(라그랑주) 좌표계에서 물리를 적용한다"는 뜻입니다. 진단량(정역학·지균 잔차, N², 포화 결손, 발산·와도, 지표 수지, TISR 위상)을 GNN 입력으로 주고, critic(해석적 진단 + 𝔤 위의 Lie–Fisher 판별)이 현실 정합성을 판정합니다. 이론 참조: Li et al., *Lie Group Machine Learning* (De Gruyter 2019). 설계 문서(한국어): https://claude.ai/artifact/7R6BJ2z27FC5LWu5m3SG18

**구현 (`geometry/composite_group.py`, `eval/physics_diag.py`, `models/pyramid_transition.py`, `models/critic.py`, `eval/stage5.py`):**
- 복합 군: Vert 이동(경계 고정, tanh 상한), Fib 생성자와 정확한 아핀 흐름 Z(dt) = e^{dt A} Z + φ₁(dt A) dt b(혼합 가족은 교환 가능 → 닫힌 아핀 부분대수, 칼럼 합 보존), Strang/Trotter 분할, Cartan 분해 ω = ω_𝔨 + ω_𝔭, Karcher 평균·군 가우시안 불변성 통계.
- 전이: 피라미드 전이의 같은 특징에서 칼럼별 수직 이동 w와 가족별 계수(복사·혼합·지표·응결, 학습 게이트)를 내고, 한 스텝을 exp(½F)·exp(w∂_η)·exp(ω)·exp(½F)로 합성합니다. 이동 헤드는 0 초기화, 가족마다 이득이 0에서 시작(혼합·응결은 straight-through 사영으로 비음수)하므로 G1은 P와 같은 모델에서 출발합니다. 생성원 크기는 L_fib로 벌점.
- 진단 입력 80채널: 층 쌍별 정역학 두께 오차와 N², 층별 지균 불균형 비(적도 ±10° 제외)·1 − q/q_s·발산·와도, 지표 지균 비·10 m 발산·와도, t2m − T₁₀₀₀. 실제 WB2 2018에서 정역학 오차 중앙값 ~1e-3, N² ~1e-4 (성층권 4e-4), 지균 비 0.1(상층)–0.25(1000 hPa)로 물리적으로 맞는 값입니다. 0.25° 격자의 극 행도 처리합니다.
- critic: 진단 critic은 관측 프레임의 EMA 통계 대비 위도 행별 동서 평균·로그 분산을 비교합니다(흐린 예측이 나쁜 점수를 받도록 분산 항 포함; 실제 2018 상태 1.29, 동서 평활 1.66, 잡음 10.6). Lie–Fisher critic은 관측 바람의 Killing 투영(같은 피라미드 사영, 스텝당 rad)과 모델 생성원 사이의 Fisher 판별(쉘·패치 레벨별, Cartan 좌표, 합동 공분산)입니다. G3는 L_phys·L_fisher를 첫 200 스텝에 걸쳐 켭니다. 앙상블 멤버 가중은 멤버 간 표준화한 점수의 softmax.
- 평가 `scripts/eval_stage5.py` + `report_stage5.py`: 쉘별 가족 활동과 불안정(N² < 0) 칼럼 조건부 혼합, 가족·이동 knock-out, TISR = 밤 반사실, Cartan 스핀/이송 에너지, critic 가중 앙상블 CRPS·spread/skill·critic–오차 순위 상관, 2018 대 2020 군 통계. P1(72h RMSE)은 `compare_graphcast.py`.
- 테스트 25개 추가(전체 83개).

**팔 (`configs/wb2_g1.yaml`·`wb2_g2.yaml`·`wb2_g3.yaml`):** G0 = 4단계 P(플래그 끔, 기존 체크포인트 그대로 로드), G1 = +Vert·Fib, G2 = +진단 입력, G3 = +critic. 같은 데이터·단계·스텝 예산. 예측 P1–P5와 판정 기준은 설계 문서 §7과 `report_stage5.py` 머리말.

### 5.1 G1 결과와 원인 분석 (`stage5_causal_lie/causes/`)

**G1 (G0 = P + Vert·Fib), 2018 61개 초기 시각, 72h RMSE의 P 대비 변화:** z500 −0.1%, t850 −0.0%, q700 +4.1%, t2m +5.6%, mslp +5.9%, u850 −4.7%; 83개 변수 평균 −0.6%(48개 개선). P1(t2m·q700·t850 3% 이상 개선) **미지지**.
- 가족 사용(1스텝 knock-out 손실 증가): 복사 +0.65%, 지표 +0.07%, 이동 +0.03%, 혼합·응결 0. 혼합 이득은 −0.054로 학습되어 비음수 사영에서 0이 되었습니다(꺼짐).
- TISR = 밤 반사실: 복사가 강한 곳 집중도 2.6배(기준 ≥ 2 통과), 효과 크기 5.6e-4(기준 3.7e-2 미달), 복원 t2m 변화 −0.003.
- 학습 없이 보정한 critic으로 멤버를 가중하면 CRPS가 오히려 나빠지고(72h z500 511 vs 496) critic 점수와 실제 오차의 순위 상관은 0.009입니다.

**원인 분석 도구 (`eval/causes.py`):** 예보 오차를 정확히 둘로 나눕니다. 자동부호화기 바닥 rec = D(E(X_k)) − X_k(어떤 전이로도 지울 수 없는 몫)와 잠재 동역학 오차 dyn = D(Z_k) − D(E(X_k)). 같은 초기 상태·같은 denoiser 잡음으로 일부를 끈 롤아웃과 초기 시각별로 짝지어 비교하고 90% bootstrap 구간을 냅니다. 31개 초기 시각, 단일 멤버, 재학습 없음.

| 지표 | S | P | P−skip | P−nodrop | G1 |
|---|---|---|---|---|---|
| z500 6h 총오차 / 자동부호화기 바닥 (m²/s²) | **264 / 254** | 357 / 361 | 431 / 451 | 280 / 271 | 357 / 356 |
| z500 72h 총오차 / 바닥 / 동역학 | 817 / **260** / 814 | 862 / 371 / 865 | 814 / 460 / **749** | 888 / 280 / 872 | 868 / 367 / 840 |
| 바닥이 설명하는 오차² 비율, 전 변수 평균 6h / 24h / 72h | 0.83 / 0.40 / 0.20 | 0.90 / 0.54 / 0.29 | 1.04 / 0.80 / 0.55 | 0.82 / 0.42 / 0.20 | 0.90 / 0.55 / 0.31 |
| denoiser 끔: z500 / t2m 72h RMSE 변화 | −2.7% / −9.5% | −6.3% / −12.2% | +1.5% / −1.9% | +1.1%* / −8.8% | −3.9% / −3.8% |
| 복합 군 끔 (G1): t850 / mslp / t2m / q700 / z500 72h | – | – | – | – | +21.2% / +10.5% / +7.8% / +2.8% / −2.2% |

(굵은 값이 가장 좋은 팔. 표의 변화는 *표시(P−nodrop z500, 구간 −0.1..+2.5%)를 빼고 모두 90% 구간이 0을 포함하지 않습니다. P−nodrop = skip dropout 0.5를 0으로 바꾼 P. 바닥 비율이 1을 넘는 경우(예: 6h t2m)는 두 시각 이력으로 만든 예측이 한 시각만 인코딩한 복원보다 정답에 가깝다는 뜻입니다.)

1. **짧은 리드 손해는 자동부호화기 때문입니다.** 6h z500 바닥이 P 361, S 254(+42%)로 P의 6h 손해(+35%)와 같은 크기이고, 6h 동역학 오차는 비슷합니다(220 vs 193). P의 repr 단계 복원 손실은 이봉 분포입니다(skip이 있으면 0.96, skip이 끊긴 표본은 2.1, S는 0.55). skip dropout(0.5)과 V-Net 자체 용량이 모두 후보라, skip dropout을 끈 **P−nodrop**을 원인 ablation으로 학습했습니다.

   **결과: 원인은 skip dropout입니다.** P−nodrop의 6h z500 바닥은 271로 S(254)에 붙었고 P(361)와의 차이의 84%가 사라졌습니다(83개 변수 평균 94%, 중앙값 92%). 변수 평균 바닥 비율도 0.82 / 0.42 / 0.20으로 S와 같습니다. 그래서 P−nodrop은 6h에서 P보다 크게 좋고(z500 −21.5%, q700 −22.1%, mslp −20.4%), 24h에서는 S와 같거나 낫습니다(t850 −4.0%, t2m −2.9%, mslp −1.7%, z500 +1.2%). 남은 V-Net 용량 몫은 z500 6h 바닥 기준 +7%입니다.

   **그러나 72h에서는 P−nodrop이 S보다 나쁩니다**(z500 +8.8%, q700 +5.8%, mslp +4.9%, t2m +3.1%; t850만 −1.4%). 바닥은 S와 같으므로 차이는 잠재 동역학입니다(z500 dyn +7.2%). 피라미드 팔 사이에서는 skip에 덜 기댈수록 72h 동역학 오차가 작습니다(P−nodrop 872 > P 865 > P−skip 749). skip dropout은 자동부호화기를 흐리게 만든 대신 거친 레벨로 정보를 보내 긴 리드 포화를 늦추는 정규화로도 작용한 것으로 해석합니다(검증 안 된 해석). 곧 피라미드 전이 자체는 아직 긴 리드에서 S의 단일 레벨 전이보다 낫지 않습니다.

2. **긴 리드 오차는 잠재 동역학이 주원인입니다**(72h 바닥 비율 0.1–0.3). P−skip의 72–120h 이득은 동역학 오차가 작아서인데(749 vs 814), 이는 상태가 거의 움직이지 않는 매끄러운 모델이라서입니다(4단계 4.1절).
3. **DIAMOND denoiser는 평균 기술을 더하지 않습니다.** 끄면 S·P·G1 모두 거의 모든 변수·리드에서 단일 멤버 RMSE가 2–12% 좋아지고(예외: G1 72h t850 +1.3%), P의 72h z500에서는 4멤버 앙상블 평균(단일 멤버 대비 −3.2%)보다도 낫습니다(−6.3%). 지금의 잔차는 잡음으로 퍼짐만 늘리고, 그래도 spread/skill은 0.25–0.4로 부족합니다.
4. **G1의 복합 군은 물리 가족이 아니라 보정항으로 쓰입니다.** 끄면 t850이 21%까지 나빠지지만(전이와 함께 적응), 혼합은 음의 이득(반확산, 곧 선명화)을 원해 꺼졌고 응결·수직 이동은 거의 0입니다. 흐린 자동부호화기를 보정하려는 신호로 해석합니다. 그래서 G1 전체는 P와 비슷합니다.
5. **critic 점수는 아직 오차를 구분하지 못합니다**(순위 상관 0.009). critic으로 학습하는 G3의 전제가 사후 검사에서 성립하지 않습니다.

**방향성:** 측정된 가장 큰 오차원은 짧은 리드의 자동부호화기 바닥과 긴 리드의 잠재 동역학·denoiser 잡음이고, 5단계의 물리 생성자는 어느 쪽도 직접 겨냥하지 않습니다. 추천 순서: (1) P−nodrop으로 자동부호화기 원인 확인 → **완료, skip dropout이 원인**이므로 이후 새 팔(6단계 H0–H2)은 skip dropout 0을 바탕으로 합니다, (2) 진단량 입력 G2는 설계대로 실행(아직 검증 안 된 유일한 5단계 요소; G1과 비교하려고 P 바탕을 유지하므로 G 팔은 자동부호화기 바닥을 그대로 안고 있고, G2 대 G1 차이만 해석합니다), (3) G3는 critic 점수가 오차를 추적할 때까지 보류, (4) 다음 후보는 결정론적 Lie 이송을 평균으로 쓰고 denoiser는 보정된 퍼짐만 내는 확률 단계입니다.


---

## 6단계 — 쉘 그래프 결합 (`stage6_shell_graph/`)

**가설 (감자, 10/31 13:18):** 쉘 사이 사슬 결합을 놓고 그래프 단위로 연결합니다. 회전의 사슬(g_s = g_{s−1}·exp δ_s)은 1단계 ablation 뒤로 이미 꺼져 있었고, 사슬이 남아 있던 곳은 모든 GNN 층의 수직 혼합(바로 위아래 쉘만 보는 `VerticalShellMixer`)과 5단계의 쉘 축 라플라시안이었습니다. 5개 쉘(지표·경계층·하부 대류권·상부 대류권·성층권)을 완전 그래프로 이으면 심층 대류(경계층 ↔ 상부 대류권), 지표 복사·플럭스처럼 이웃을 건너뛰는 연직 결합을 직접 표현할 수 있고, 5단계 G1에서 사슬 혼합이 꺼진 것도 국소 사슬 구조가 맞지 않았기 때문인지 가릴 수 있습니다.

**구현 (`models/layers.py`, `models/pyramid_transition.py`, `eval/causes.py`):**
- `ShellGraphMixer` (`model.gnn.vertical: graph`): 모든 쉘 쌍의 메시지 전달. 가중치 a_ss' = softmax(B_ss' + ⟨q(h_s), k(h_s')⟩/√d), B는 사슬 이웃을 선호하도록 초기화(비사슬 간선 몫 0.11)하고 q는 0에서 시작합니다. 받는 쉘은 다른 쉘 상태의 가중 평균과 쉘 쌍 임베딩의 가중 평균(어디서 온 정보인지)을 받습니다. 같은 노드의 쉘은 접평면을 공유하므로 벡터 이송이 필요 없고 SO(3) 등변성이 유지됩니다. 인코더·디코더·V-Net·전이·denoiser의 사슬 혼합기 13개를 모두 바꿉니다(파라미터 +9%).
- 생성원 그래프 결합 (`model.lie.shell_coupling: graph`, 피라미드 전이): 모든 패치·레벨에서 ω_s = δ_s + Σ_{s'≠s} A_ss' δ_s'. A = 0이면 독립 쉘, 엄격 하삼각 1이면 1단계 사슬의 1차(BCH) 근사입니다(테스트로 확인). 관측 바람의 생성원(5단계 critic)에는 적용하지 않습니다.
- 원인 분석: `nonlocal`(혼합기를 사슬 간선으로 제한), `shell_graph`(A = 0) knock-out, 학습된 그래프(사전 B, 상태 의존 평균 가중치, 비사슬 몫, A) 요약을 `eval_causes.py`·`report_causes.py`에 추가했습니다.
- 물리 수준의 그래프 라플라시안 혼합(L = D − W, W ≥ 0)은 H 팔에 복합 군이 없어 보류했습니다.

**팔:** H0 = P−nodrop(사슬 혼합, 독립 쉘; skip dropout을 끈 4단계 P), H1 = H0 + 쉘 그래프 혼합기(`configs/wb2_h1.yaml`), H2 = H1 + 생성원 그래프 결합(`configs/wb2_h2.yaml`). 같은 데이터·단계·스텝 예산.

**예측(사전 등록):** (1) H1은 H0보다 72h 단일 멤버 RMSE가 연직 교환 변수(q700, t850, t2m)에서 2% 이상 낮고 90% 구간이 0을 포함하지 않는다. (2) 비국소 간선이 쓰인다: 상태 의존 비사슬 몫이 사전(0.11)보다 크고, `nonlocal` knock-out이 RMSE를 올린다. (3) 가장 큰 비사슬 가중치는 경계층·지표와 상부 대류권 사이(심층 대류)에 있다. (4) H2의 A는 0이 아니고 `shell_graph` knock-out이 RMSE를 올린다; A가 사슬(하삼각 양수)에 가까운지, 순압 결합(모두 양수)에 가까운지 기록한다.
---

## 7단계 — 예측 게임과 GNN feature 제어 (`stage7_prediction_game/`)

**가설 (감자, 10/31 14:06):** Lie 구조에 매몰되지 말고, 세계 모델이 물리를 이해해 미래를 예측하게 만드는 데 집중합니다. 예측을 하나의 게임으로 보고(Ha & Schmidhuber 2018 World Models, Freeman·Metz·Ha 2019 observational dropout), (1) 물리 예측 구조를 먼저 학습해 고정하고, (2) 마스킹된 데이터를 예측하는 게임에서 GNN feature를 조절하는 제어만 보상으로 학습한 뒤, (3) 전체 모델을 시험합니다. 설계 문서: [7단계 설계: 예측 게임과 GNN feature 제어](https://claude.ai/artifact/H8az4WzNqJvUuvWYYGz1LU).

**구현 (`models/feature_control.py`, `train/prediction_game.py`):**
- 제어기(`FeatureController`, 8.9k 파라미터): 쉘 × 12개 패치마다 잠재 스칼라·벡터 크기 평균, 디코딩한 현재 상태의 물리 진단량(5단계 모듈의 7개 가족을 쉘별 평균), 마스크 비율, 쉘·패치 정체를 보고, 잠재 스칼라에 저차수 FiLM h ← h + 0.1·(h·Uγ + Uβ)(r = 4, U 고정 직교 기저)를 겁니다. 스칼라만 바꾸므로 SO(3) 등변성은 유지되고, 마지막 층이 0에서 시작해 학습 전에는 고정 모델과 똑같습니다. 세계 모델의 하위 모듈로 등록하지 않아 체크포인트는 바뀌지 않습니다(`controlled(model, ctrl)`).
- 게임: 24h(4스텝) 판마다 변수 하나(모든 층) 또는 표면 묶음, 쉘 하나, 지구 면적 25%의 원형 지역 중 하나를 가리고 기후값(0)으로 채웁니다. 점수 = −Σ_k [0.5·가려진 점의 MSE + 0.5·전체 MSE](GraphCast 가중). 세계 모델은 denoiser를 끈 결정론 모드입니다. 시간 마스킹(확률적으로 실제 관측을 다시 보는 observational dropout)은 구현했지만 1차에서는 끕니다.
- 학습: `rl` = 상태마다 행동 4개(가우스 정책, σ = 0.5)로 그룹 정규화 이점의 정책 기울기(2단계 GRPO 방식); `grad` = 같은 제어기를 고정 모델을 통한 정확한 기울기로 학습(대조군). 150회 갱신, 같은 창·같은 마스크 순서, 10회마다 이어 하기 저장.
- 원인 분석: `eval_causes.py --controller --deterministic`에 `no_control`(고정 모델), `random_control`(같은 행동을 섞인 쉘·패치 위치에), `no_diag`(진단량 입력 0) knock-out을 추가했고, `eval_game.py`는 같은 knock-out을 마스킹 게임에서 잽니다.

**팔:** 기본 모델 = P−nodrop(2015–17 학습, 고정). C0 = 행동 0, C1 = RL 제어기, C1-grad = 기울기 대조군, C1-open = 상태를 보지 않는 상수 행동, C1-random·C1-nodiag = C1의 평가 시 knock-out. 게임 학습 2018–19, 시험 2020(31개 초기 시각). 2차 후보: C1-nomask(마스킹 없는 게임), 입력 feature 게이트 행동.

**예측(사전 등록):** Q1 C1이 2020 마스킹 게임에서 가려진 변수의 24h 오차를 C0보다 3% 이상 줄이고 90% 구간이 0을 포함하지 않는다. Q2 C1의 이득이 C1-random·C1-open보다 크다. Q3 마스킹 없는 보통 예보에서 6–72h 어느 리드에서도 0.5% 넘게 나빠지지 않는다. Q4 C1-nodiag가 이득의 절반 이상을 잃는다. Q5(기록) C1 대 C1-grad. Q1과 Q2가 모두 성립하면 지지로 판정합니다.

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
# 3단계 2차: E2 = wb2_theory_hybrid.yaml, D = wb2_theory_phys.yaml (같은 --init-from), 2020 불변성은 DATA2020에 같은 평가
python scripts/eval_transfer.py --checkpoint runs/wb2_theory_hybrid/checkpoint.pt --data-dir DATA2020 --stride 4 \
    --min-gap-days 7 --out runs/wb2_theory_hybrid/transfer_2020.json
python scripts/report_theory.py C=runs/wb2_theory_lie/transfer.json E2=runs/wb2_theory_hybrid/transfer.json \
    D=runs/wb2_theory_phys/transfer.json --invariance C=runs/wb2_theory_lie/transfer_2020.json \
    E2=runs/wb2_theory_hybrid/transfer_2020.json --out-dir docs/results/stage3_theory
# 4단계
python scripts/train.py --config configs/wb2_pyramid.yaml --data-dir DATA              # P (P−skip: wb2_pyramid_noskip.yaml)
python scripts/compare_graphcast.py --checkpoint runs/wb2_pyramid/checkpoint.pt --data-dir DATA --init-stride 3
python scripts/eval_pyramid.py --checkpoint runs/wb2_pyramid/checkpoint.pt --data-dir DATA --stride 4
python scripts/report_pyramid.py S=runs/independent/compare_graphcast.json P=runs/wb2_pyramid/compare_graphcast.json \
    "P-skip=runs/wb2_pyramid_noskip/compare_graphcast.json" --pyramid P=runs/wb2_pyramid/pyramid_eval.json \
    "P-skip=runs/wb2_pyramid_noskip/pyramid_eval.json" --out-dir docs/results/stage4_pyramid
# 7단계 (기본 = P−nodrop 고정; DATA2020 = 2017-12 .. 2021-01 ERA5)
python scripts/play_game.py --checkpoint runs/wb2_pyramid_nodrop/checkpoint.pt --config configs/game_wb2.yaml \
    --data-dir DATA2020 --mode rl --out runs/game_c1        # C1-grad: --mode grad, C1-open: --set control.use_state=false
python scripts/eval_game.py C1=runs/game_c1/controller.pt --checkpoint runs/wb2_pyramid_nodrop/checkpoint.pt \
    --config configs/game_wb2.yaml --data-dir DATA2020 --out runs/game_c1/game_eval.json
python scripts/eval_causes.py --checkpoint runs/wb2_pyramid_nodrop/checkpoint.pt --controller runs/game_c1/controller.pt \
    --data-config configs/game_wb2.yaml --data-dir DATA2020 --deterministic --variants no_control random_control no_diag \
    --out runs/game_c1/causes_2020.json
python scripts/report_game.py --game runs/game_*/game_eval.json --forecast C1=runs/game_c1/causes_2020.json \
    --out-dir docs/results/stage7_prediction_game
# 6단계 (H0 = P−nodrop, H1, H2)
python scripts/train.py --config configs/wb2_h1.yaml --data-dir DATA        # H2: wb2_h2.yaml, H0: wb2_pyramid_nodrop.yaml
python scripts/eval_causes.py --checkpoint runs/wb2_h1/checkpoint.pt --data-dir DATA --variants no_denoiser nonlocal
python scripts/report_causes.py H0=runs/wb2_pyramid_nodrop/causes.json H1=runs/wb2_h1/causes.json \
    H2=runs/wb2_h2/causes.json --out-dir docs/results/stage6_shell_graph
# 5단계 (G1–G3 각각; G0 = 4단계 P 체크포인트)
python scripts/train.py --config configs/wb2_g1.yaml --data-dir DATA
python scripts/compare_graphcast.py --checkpoint runs/wb2_g1/checkpoint.pt --data-dir DATA --init-stride 3
python scripts/eval_stage5.py --checkpoint runs/wb2_g1/checkpoint.pt --data-dir DATA
python scripts/report_stage5.py G0=runs/wb2_pyramid/compare_graphcast.json G1=runs/wb2_g1/compare_graphcast.json \
    G2=runs/wb2_g2/compare_graphcast.json G3=runs/wb2_g3/compare_graphcast.json \
    --eval G0=runs/wb2_pyramid/stage5_eval.json G1=runs/wb2_g1/stage5_eval.json G2=... G3=... \
    --out-dir docs/results/stage5_causal_lie
```
