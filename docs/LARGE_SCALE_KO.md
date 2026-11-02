# 대용량 학습 가이드 (Lustre hourly ERA5 0.25°, GPU 96GB 1장)

CPU 실험(WB2 64×32)과 같은 모델과 같은 평가를, Lustre의 hourly 0.25° ERA5(1979–2018, 약 80TB, 2019–2025 추가 예정)로 학습하는 방법입니다.

## 1. 무엇이 바뀌었나

| 부분 | CPU 실험 | 대용량 버전 |
|---|---|---|
| 데이터 읽기 | zarr 전체를 메모리에 올림 | `HourlyArchive`: 파일별 시간 색인을 한 번 만들고, 필요한 6시간 프레임만 그 파일에서 읽음. 파생 복사본 없음 |
| 해상도 | 5.6° | 0.25° 원본, 또는 읽으면서 1°/0.5°로 면적 보존 평균(`coarsen: 4`/`2`) |
| 강수 | 6h 값 제공 | hourly `tp` 6개를 합쳐 `total_precipitation_6hr` 생성 |
| 정규화 통계 | 64개 window | 무작위 window 표본을 worker 여러 개로 읽어 계산하고 `stats_path`에 저장, 재사용 |
| 학습 | fp32, CPU | bf16 autocast (SO(3) 대수·군 작용·손실은 fp32 유지), gradient checkpointing, gradient accumulation, 다중 worker 로딩 |
| 중단 대비 | stage 끝에 저장 | `save_every` step마다 optimizer·scheduler까지 저장, `--resume`으로 그 step부터 재개 |
| 해상도 이동 | - | `--init-from`: 1° 가중치를 0.25° 모델로 이어받기 (그래프만 새로 생성) |
| 평가 기후값 | 64×32 로컬 | WB2 0.25° 기후값을 GCS에서 필요한 날짜만 읽음 (`climatology_load: false`, 1°는 `climatology_coarsen: 4`) |

## 2. 데이터 준비

`configs/era5_lustre_hourly.yaml`의 경로를 채웁니다.

- 기압면 파일 (z, q, t, u, v, 13개 층)과 지표 파일 (u10, v10, mslp, t2m, tp, tisr 등)은 glob으로 지정합니다. `**`도 됩니다.
- **정적 필드가 필요합니다.** GraphCast 입력에는 지표 지오포텐셜과 육지-바다 마스크가 들어가는데, 받은 목록에는 없습니다. ERA5 invariants 파일을 `rename: {z: geopotential_at_surface}`로 추가하세요.
- w(연직속도)가 없으므로 task는 `graphcast_13_no_w`입니다.
- 파일 형식은 netCDF 기준이고, `engine`을 바꾸면 GRIB(cfgrib), h5netcdf, zarr도 읽습니다. CDS 형식(`valid_time`, `pressure_level`, 위도 내림차순, Pa 단위 층, `expver`)은 자동으로 맞춥니다.

먼저 점검합니다. 처음 한 번은 모든 파일의 헤더를 읽어 색인을 만들고 `index_cache`에 저장합니다. 2019–2025 파일이 추가되면 새 파일만 읽습니다.

```bash
python scripts/inventory.py --archive configs/era5_lustre_hourly.yaml --task graphcast_13_no_w
```

변수별 시간 범위, 빠진 변수, 모든 target이 갖춰진 프레임 수를 보여주고, 한 프레임을 실제로 읽어 NaN을 확인합니다.

**저장 공간:** 기본 설정은 아무것도 복사하지 않습니다. 대신 1° 학습도 매번 0.25° 원본을 읽고 평균하므로 I/O는 0.25°와 같습니다. 0.25° 표본 하나(입력 2 + target 1–2 프레임)가 약 1–1.3GB입니다. I/O가 병목이면 1° 6시간 캐시(40년, fp16 약 0.5TB)를 만드는 방법이 있습니다. 필요하면 말씀해 주세요.

## 3. GPU 메모리부터 측정

데이터 없이 실제 크기의 가짜 배치로 stage마다 step 시간과 최대 메모리를 잽니다. 긴 학습 전에 꼭 먼저 돌려주세요.

```bash
python scripts/benchmark_step.py --config configs/p2_1deg_96gb.yaml --device cuda
python scripts/benchmark_step.py --config configs/p2_025deg_96gb.yaml --device cuda
# 크기를 바꿔 보기
python scripts/benchmark_step.py --config configs/p2_025deg_96gb.yaml --device cuda \
    --set model.gnn.hidden_dim=192 train.batch_size=1 train.engine.grad_checkpointing=true
```

출력의 `s/step`과 `peak GB`로 고릅니다.

- 메모리가 넘치면 `grad_checkpointing: true`, `batch_size` 감소(`accum_steps`로 보충), `hidden_dim` 감소, `latent_splits` 감소 순서로 줄입니다.
- 학습 시간은 대략 Σ(stage step 수 × s/step)입니다. 설정 파일의 step 수는 시작점이므로 측정값을 보고 조정하세요.
- 같은 출력에 평가 시간(init 하나당 초)도 나옵니다.

## 4. 학습 순서

```bash
# 1° 사전학습 (정규화 통계는 처음 한 번 계산되어 runs/p2_1deg/stats.json에 저장)
python scripts/train.py --config configs/p2_1deg_96gb.yaml --device cuda --resume

# 0.25° 미세조정: 1° 가중치를 이어받음
python scripts/train.py --config configs/p2_025deg_96gb.yaml --device cuda \
    --init-from runs/p2_1deg/checkpoint.pt --resume
```

`--resume`은 처음에는 아무 일도 하지 않고, 중간에 끊긴 뒤 다시 실행하면 마지막 저장 step부터 이어갑니다. 같은 명령을 그대로 다시 실행하면 됩니다.

설정 요약:

| | 1° (`p2_1deg_96gb`) | 0.25° (`p2_025deg_96gb`) |
|---|---|---|
| 격자 | 181 × 360 (원본을 4배 평균) | 721 × 1440 |
| 메시 | M5 10,242 노드 (잠재도 M5) | 인코더·디코더 M6 40,962 노드, 잠재 M5 |
| 폭 | hidden 256, 잠재 64 스칼라 + 16 벡터 | 같음 (가중치 이어받기) |
| 배치 | 2 × accumulation 2 | 1 × accumulation 4, checkpointing |
| step | repr 20k, dynamics 60k (2-step), stochastic 30k | 5k, 20k, 10k |
| 학습 기간 | 1979–2017 | 1979–2017 |
| 평가 | 2018, 6–240h, 8 member | 2018, 6–240h, 4 member |

## 5. GraphCast와 비교

WB2의 GraphCast 2018 0.25° 예측을 GCS에서 바로 읽습니다. 1° 모델은 `--coarsen 4`로 같은 격자에 맞춥니다.

```bash
python scripts/compare_graphcast.py --checkpoint runs/p2_1deg/checkpoint.pt \
    --graphcast gs://weatherbench2/datasets/graphcast/2018/date_range_2017-11-16_2019-02-01_12_hours.zarr \
    --coarsen 4 --init-stride 4 --lead-steps 40
python scripts/report_comparison.py runs/p2_1deg/compare_graphcast.json
```

0.25° 예측은 init 하나에 40 lead × 70 채널이 약 11GB라서 GCS에서 읽는 데 시간이 걸립니다. 처음에는 `--init-stride`로 init 수를 줄이세요.

## 6. 아직 없는 것

- 여러 GPU 분산 학습(DDP)은 아직 없습니다. 1장 기준입니다.
- 혼합정밀은 bf16만 지원합니다(fp16 + loss scaling 없음).
- 1° 6시간 캐시를 쓰는 스크립트는 저장 공간 확인 후 추가합니다.
- 큰 설정의 메모리와 속도는 이 저장소 환경(CPU)에서 측정하지 못했습니다. 위 benchmark로 GPU에서 먼저 확인해야 합니다.
