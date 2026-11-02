# Geometric Stochastic Climate World Model: 코드 설계 (v0.1)

연구 설계 문서 `Climate_World_Model_Research_Design_KO_20261030`의 1차 구현입니다.
가설과 결과는 모두 **검증 대상**이며, 이 코드는 그 검증을 위한 실험 틀입니다.

## 1. 전체 구조

```
ERA5 (GraphCast task)                        data/graphcast_mapping.py, era5_lazy_loader.py
  X[t-6h], X[t] + forcing(t+6h) + static
        │  raw -> shell 분할 (지표 shell + 압력층 shell),  wind (u,v) -> 접벡터
        ▼
Shell-GNN encoder E_phi  (grid -> mesh -> latent mesh, down-scaling)     models/shell_gnn_encoder.py
        │  Z_t = (h [B,S,N,Cs], V [B,S,N,Cv,3])  : shell x node 의 스칼라 + 접벡터
        ▼
Lie generator  xi_t in so(3)^{shell x patch}                              models/lie_generator.py
        │  Zbar_{t+1} = Phi_{exp(dt xi)}(Z_t)   (semi-Lagrangian SO(3) 작용)   geometry/group_actions.py
        ▼
Residual transition (DIAMOND)  r ~ p(r | Zbar, Z_{t-1:t}, c_{t+1})        models/stochastic_transition.py
        │  Z_{t+1} = Exp_Zbar(r)   (EDM denoiser, Euler/Heun sampler, 3 steps)
        ▼
Shell-GNN decoder D_psi  (latent mesh -> mesh -> grid, up-scaling)       models/shell_gnn_decoder.py
        │  Xhat_{t+1}  (GraphCast target 변수, lat-lon 격자)
        ▼
autoregressive latent rollout (6h x L)  ->  GraphCast 평가 (RMSE 식20, ACC 식29)   eval/graphcast_metrics.py
```

## 2. GraphCast 입출력 정렬

`weathernext/weathernext1_graph/graphcast.py`의 `TaskConfig`를 그대로 옮겼습니다.

| task 이름 | 원본 | 비고 |
|---|---|---|
| `graphcast` | `TASK` | 37 levels, 0.25° 논문 모델 |
| `graphcast_13` | `TASK_13` | 13 levels (A안: w 확보 시 정확한 정렬) |
| `graphcast_operational` | `TASK_13_PRECIP_OUT` | 강수 입력 없음 |
| `graphcast_13_no_w` | B안 | 우리 ERA5에 w가 없으므로 기본값 |

- 입력: `X[t-6h], X[t]` (target 변수 [+강수]) + forcing (`toa_incident_solar_radiation`, year/day progress sin·cos) + static (`geopotential_at_surface`, `land_sea_mask`)
- forcing: target 시각 `t+6h`의 forcing (DIAMOND의 action 자리에 들어감)
- 출력: `X[t+6h]` = 지표 5개 + 대기 5(6)개 x 13 levels = 70 (83) 채널
- 손실: GraphCast와 같은 위도·기압 가중 MSE와 변수 가중치 (`train/losses.py`)
- 정규화: 스칼라는 (x-mean)/std, 바람 (u,v)는 같은 scale·평균 0 (벡터 방향 보존)

## 3. 구면 분할: shell 과 patch

- **shell (수직)**: 지표 shell + `boundary_layer [1000,925,850]`, `lower_troposphere [700,600,500]`,
  `upper_troposphere [400,300,250,200]`, `stratosphere [150,100,50]` (`geometry/shells.py`).
  shell마다 입력 embedding은 따로, GNN 가중치는 공유 + shell embedding, shell 사이는 수직 message passing.
- **mesh (수평)**: GraphCast와 같은 icosahedral multimesh (`geometry/spherical_mesh.py`).
  `mesh_splits` 단계에서 처리하고 `latent_splits` 단계(더 거친 mesh)로 pooling 하여 latent를 만듭니다.
- **patch**: 거친 icosphere(`patch_splits`)의 Voronoi 영역. (shell, patch) 가 Lie 전이의 기본 단위입니다.

## 4. Lie group / Lie algebra 의 역할

- so(3) ≅ 구면의 Killing 벡터장: ω ∈ so(3) 는 속도장 u(p) = ω × p (강체 회전)를 만듭니다.
  - ω 의 p 에 수직인 성분은 점을 이동(이류), p 방향 성분은 접평면을 회전(바람 벡터 회전).
- generator 는 latent 접벡터로부터 속도장을 제안하고, patch 마다 최소제곱 Killing 장
  ω_P = argmin |ω × p − u|² 로 **Lie algebra 에 사영**합니다 (`SpherePatches.killing_projection`).
- node 별 ω 는 algebra 에서 섞고(군 원소는 평균할 수 없음) exp 로 SO(3) 원소를 만든 뒤
  semi-Lagrangian 작용 Φ_g(h)(p) = h(g⁻¹p), Φ_g(V)(p) = g V(g⁻¹p) 로 latent 를 수송합니다.
  - g = e 이면 정확히 항등, ω 에 대해 미분 가능, 전역 회전에 대해 등변 (테스트로 확인).
- Lie 부분이 표현할 수 없는 것 (발산, 변형, 가열·응결, 계의 분열·병합)은 residual / 확률 전이가 맡습니다.

## 5. "Liquid centipede robot" 가설과 코드 대응

| 로봇 논문의 요소 | 기상장 해석 | 코드 |
|---|---|---|
| Liquid robot: 강체처럼 이동 | shell 의 회전(pose) | `LieGenerator` 의 so(3) per (shell, patch) |
| Liquid robot: 변형 | 비강체 변형, 발산 | `ResidualDynamics` / diffusion residual |
| Liquid robot: 분열·병합·포획 | 저기압 분열·병합, blocking 같은 위상 변화 | DIAMOND 확률 residual (앙상블) |
| 입자 껍질 (액체 누출 방지) | 질량·수분·에너지 보존 제약 | Stage 4 `L_phys` (예정) |
| Centipede: 모듈 + 3자유도 유연 관절 | shell 사슬, 관절 회전 = 연직 시어 | `shell_coupling: chain`, g_s = g_{s-1}·exp(gate·δ_s) |
| 관절 토션 스프링 | 연직 결합 강성 (thermal wind 류) | `joint` 손실 = Σ\|gate·δ_s\|² |
| 유연 / 강체 연결 전환 | 경압(baroclinic) / 순압(barotropic) | 학습되는 gate ∈ (0, 1) |
| 지형에 수동 적응하는 바퀴 | 지형·지표 강제에 대한 하층 반응 | static 입력, 지표 shell 이 사슬의 기준 |
| action (제어 입력) | 외부 강제력 TISR, 지형, (SST) | forcing 을 DIAMOND action 자리에 조건화 |

검증 방법: 같은 예산에서 `shell_coupling: independent` 와 `chain` 을 비교 (ablation),
GraphCast RMSE/ACC 와 보존량·스펙트럼 진단을 함께 봅니다.
한계: 대기에는 policy/액추에이터가 없고, 강체·관절 묘사는 대규모 저주파 운동만 잡습니다.

## 6. DIAMOND 에서 가져온 것과 바꾼 것

| DIAMOND | 이 코드 |
|---|---|
| 다음 frame x_{t+1} | latent residual r = (Z_{t+1} − Zbar_{t+1}) / scale |
| 과거 frame k 개 | 과거 latent Z_{t-1}, Z_t 와 Zbar_{t+1} |
| action embedding | latent node 의 forcing |
| UNet + AdaGroupNorm | Shell-GNN + AdaLayerNorm (noise embedding) |
| EDM 전처리 (c_in, c_out, c_skip, c_noise), offset noise | 동일 |
| Euler/Heun sampler, Karras sigma, 3 steps | 동일 (EDM 처럼 sigma_max·N(0,I) 에서 시작) |
| 학습 시 denoised frame 을 다음 조건으로 사용 | 동일 (latent rollout) |
| 픽셀 양자화 | 없음 |

## 7. 전이 모드와 ablation (설계서 10장)

| `transition` | 설명 | 설계서 |
|---|---|---|
| `euclidean` | Zbar = Z_t + 결정론적 residual | B2/B3 |
| `lie` | Lie 구조 전이만 | B4 |
| `lie_residual` | Lie + 결정론적 residual | B5 |
| `lie_diffusion` | Lie + DIAMOND residual | B6 |
| `euclidean_diffusion` | Euclidean latent diffusion 기준선 | H3 비교 |

`mesh.break_symmetry: false` 로 두면 encoder/processor 가 정확히 SO(3) 등변이 되어 H4 진단에 씁니다.

## 8. 학습 단계 (설계서 7장)

1. `repr`: Shell-GNN autoencoder, L_rec
2. `dynamics`: 결정론적 전이 + multi-step latent rollout,
   L_forecast + λ L_latent + λ L_rec + λ L_geo(|ω|²) + λ L_joint
3. `stochastic`: residual 의 EDM denoising 손실 (encoder/decoder/Lie 고정)

## 9. 평가 (GraphCast 기준)

- RMSE 식(20) (초기시각마다 sqrt 후 평균, WeatherBench 관례), 식(21) 전통적 RMSE
- ACC 식(29), 기후값은 validity time 의 day-of-year x hour (WeatherBench-2 climatology zarr 지원)
- skill score: (RMSE_A − RMSE_B)/RMSE_B, (ACC_A − ACC_B)/(1 − ACC_B)
- 물리 단위, float32, 위도 면적 가중 (극점 셀 처리 포함)
- 앙상블: fair CRPS, ensemble-mean RMSE, spread/skill
- 기준선: persistence, climatology (B0)

## 10. 알려진 한계 / 다음 단계

- decoder 에 GraphCast 의 grid skip connection 이 없음 (순수 latent 디코딩). 고해상도에서 흐려질 수 있음.
- latent manifold 는 fibre-wise 선형 (Exp = 더하기 + 접평면 사영). 진짜 다양체 구조는 base S² 와 군 작용에만 있음.
- Killing 사영은 작은 patch 에서 spin 성분이 불량조건이라 spin 을 별도 head 로 둠.
- Stage 4 물리 손실, 0.25° 분산 학습(P4)은 아직 없음.
