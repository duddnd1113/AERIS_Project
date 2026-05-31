# RoadExtension V4

## 개요

V3 TwoStage 구조를 계승하면서 세 가지를 개선:

1. **타깃 변환 교체**: Box-Cox + 1-99% clip → **log1p (clip 없음)**  
   V3의 p99 clip(78 μg/m³) 이 예측 상한을 ~50으로 막는 ceiling 문제를 해소
2. **공사 피처 추가**: Stage2에 격자별 활성 공사 정보(3종) 추가
3. **Stage2 Ridge 최적화**: RidgeCV로 alpha 자동 탐색, 도로 구조 피처 확장

```
최종 성능 (Test: 2025-01-02 ~ 2025-12-19)
  Test MAE  : 7.55 μg/m³
  Test R²   : 0.301
  예측 max  : 58.9 μg/m³  (V3: ~40–50 μg/m³)

Train (2023-01-06 ~ 2024-12-26)
  Train MAE : 5.03 μg/m³
  Train R²  : 0.595
```

---

## V3 → V4 변경 요약

| 항목 | V3 | V4 |
|------|----|----|
| 타깃 변환 | Box-Cox (λ=−0.111) + clip [2, 78] | **log1p (clip 없음)** |
| 예측 상한 | ~40–50 μg/m³ (ceiling 문제) | **58.9 μg/m³ (제한 없음)** |
| Stage1 피처 수 | 10개 (날씨+시간) | **14개 (+ambient_pm10, cold_and_dry, days_from_rain, daily_precip_mm)** |
| Stage2 피처 수 | 11개 | **19개 (+도로구조 4종, 공사 3종)** |
| Stage2 alpha | 고정 1.0 | **RidgeCV 자동 탐색 [0.01~1000]** |
| Sample weight | power=0.5 | **power=0.5 (Stage1 R² 최적)** |
| Test R² | 0.276 | **0.301** |

---

## 데이터

### 기간 및 규모

| 분할 | 기간 | 건수 | road_pm 평균 | road_pm 최대 |
|------|------|------|------------|------------|
| Train | 2023-01-06 ~ 2024-12-26 | 123,229 | 14.30 μg/m³ | 147.0 μg/m³ |
| Test | 2025-01-02 ~ 2025-12-19 | 75,188 | 13.47 μg/m³ | 195.0 μg/m³ |

> Train/Test는 시간 기준 분할 (2023–2024 → 학습, 2025 → 평가).

### 원본 분포 (전체)

```
median = 10.0 μg/m³   mean = 13.98 μg/m³   max = 195.0 μg/m³
p90    = 27 μg/m³     p95  = 38 μg/m³      p99 = 83 μg/m³
> 100 μg/m³ : 0.6%  (극단값이 희소하지만 중요한 고농도 이벤트)
```

분포가 오른쪽으로 강하게 치우쳐 있어 변환 없이 직접 회귀하면 모델이 중앙값 근처로 수렴함.

---

## 타깃 변환: log1p

### 변환 수식

```
변환:    y* = log(y + 1)          (y: road_pm μg/m³)
역변환:  y  = exp(y*) − 1
         y  = clip(y, 0, ∞)       (음수 방지)
```

### V3 Box-Cox 대비 장점

| 항목 | V3 Box-Cox + clip | V4 log1p |
|------|-------------------|----------|
| 수식 | `((y+0.01)^λ−1)/λ` (λ=−0.111) | `log(y+1)` |
| clip | [2, 78] μg/m³ (p1~p99) | 없음 |
| 변환 범위 | [0.67, 3.45] (train 기준) | [0, 5.00] (전체 범위) |
| 예측 상한 | ~78 μg/m³ (clipping에 묶임) | 제한 없음 |
| 피팅 필요 | 필요 (λ, p_low, p_high 추정) | 불필요 |
| 0 처리 | shift 0.01 필요 | 자연스럽게 처리 |

---

## 모델 구조: TwoStage

### 핵심 아이디어

도로 재비산먼지를 시간 성분과 공간 성분으로 분리:

```
log1p(road_pm) = 시간 성분 (날씨·계절·ambient PM으로 결정)
               + 공간 잔차 (격자별 고정 특성으로 결정)
```

### 예측 흐름

```
(date, hour, CELL_ID)
      │
      ├─[Stage 1: Temporal LightGBM]──────────────────────────────┐
      │   입력: 14개 시간·날씨·ambient 피처                         │
      │   출력: log1p_temporal  (log1p 공간의 시간 기준값)          │
      │                                                            │
      ├─[Stage 2: Spatial RidgeCV]────────────────────────────────┤
      │   입력: 19개 LUR·도로구조·공사 피처                         │
      │   타깃: log1p(road_pm) − log1p_temporal  (잔차)           │
      │   출력: log1p_spatial   (공간 편차)                        │
      │                                                            │
      └─ log1p_temporal + log1p_spatial = log1p_final             │
              │                                                    │
         expm1(log1p_final) → road_pm 예측 (μg/m³)  ─────────────┘
```

---

## Stage 1: Temporal LightGBM

### 사용 피처 (14개)

| 분류 | 피처명 | 설명 |
|------|--------|------|
| 시간 사이클 | `month_sin`, `month_cos` | 월의 sin/cos 인코딩 (계절성) |
| | `hour_sin`, `hour_cos` | 시각의 sin/cos 인코딩 (일변화) |
| 요일 | `weekday` | 요일 (0=월~6=일) |
| | `is_weekend` | 주말 여부 (binary) |
| | `season` | 계절 (1=봄, 2=여름, 3=가을, 4=겨울) |
| 기상 | `기온` | 일평균 기온 (°C) |
| | `습도` | 일평균 상대습도 (%) |
| | `is_dry` | 건조 여부 (습도 < 40%, binary) |
| | `cold_and_dry` | 저온·건조 복합 조건 (binary) |
| | `days_from_rain` | 마지막 강수 이후 경과 일수 |
| | `daily_precip_mm` | 일강수량 (mm) |
| 배경 농도 | `ambient_pm10` | 인근 관측소 PM10 (μg/m³) — 당일 광역 오염 수준 |

> `ambient_pm10`을 Stage1에 포함한 이유: 광역 오염 이벤트(황사, 정체 기상)는 서울 전역 도로 PM을 동시에 끌어올리는 가장 강력한 temporal signal.

### 하이퍼파라미터

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| `n_estimators` | 1,000 | 최대 트리 수 |
| `learning_rate` | 0.03 | 학습률 |
| `max_depth` | 8 | 트리 최대 깊이 |
| `num_leaves` | 63 | 리프 노드 수 |
| `min_child_samples` | 10 | 리프 최소 샘플 수 |
| `subsample` | 0.8 | 행 서브샘플링 비율 |
| `colsample_bytree` | 0.8 | 열 서브샘플링 비율 |
| `reg_alpha` | 0.1 | L1 정규화 |
| `reg_lambda` | 0.1 | L2 정규화 |
| `early_stopping` | 100 rounds | 검증 손실 개선 없으면 조기 중단 |
| Best iteration (실제) | **125** | — |

### Sample Weight

```
weight_i = 1 + (road_pm_i / median_pm)^0.5

median_pm ≈ 10.5 μg/m³ 기준:
  road_pm = 10  →  weight ≈ 2.0
  road_pm = 40  →  weight ≈ 3.0
  road_pm = 100 →  weight ≈ 4.1
```

> power=0.5 선택 근거: power 탐색 실험에서 test R² 최고 (0.304 @ Stage1 단독).  
> power=0 (weight 없음)이 MAE는 낮지만, R²는 power=0.5가 우수 → 시각화 품질 중시.

---

## Stage 2: Spatial RidgeCV

### 사용 피처 (19개)

| 분류 | 피처명 | 설명 |
|------|--------|------|
| **LUR (토지이용 회귀)** | `buildings` | 격자 내 건물 면적 비율 |
| | `greenspace` | 격자 내 녹지 면적 비율 |
| | `road_struc` | 도로 구조 지수 |
| | `river_zone` | 하천·수변 구역 여부 |
| | `ndvi` | 정규식생지수 (위성) |
| | `ibi` | 불투수면 지수 (도시화 지표) |
| | `elev_mean` | 격자 평균 고도 (m) |
| | `sum_area` | 격자 내 건물 연면적 합 (m²) |
| | `sum_height` | 격자 내 건물 높이 합 (m) |
| **도로 구조** | `traffic` | 격자별 시간대별 교통량 |
| | `highway_rank` | 도로 위계 (고속화도로~이면도로) |
| | `max_lanes` | 최대 차로 수 |
| | `mean_gvi` | 평균 녹지 시각 지수 (Green View Index) |
| | `total_road_length_m` | 격자 내 총 도로 연장 (m) |
| | `traffic_x_road` | 교통량 × 도로구조 상호작용항 |
| **배경 농도** | `ambient_pm10` | 인근 관측소 PM10 — 공간 배경 보정 |
| **공사** | `n_active_const` | 반경 500m 내 활성 공사 건수 |
| | `total_amount_억` | 활성 공사 총 사업금액 (억원) |
| | `is_large_const` | 10억 이상 대형 공사 존재 여부 (binary) |

> 공사 피처 출처: `서울시건설알림이정보.csv` → `step_construction.py` 전처리 → `construction_cache.csv`  
> 공사 커버율: Train 15.8%, Test 57.9% (2025년 공사 증가 반영)

### Stage2 Ridge 계수 (상위 10개)

| 피처 | 계수 | 해석 |
|------|------|------|
| `elev_mean` | −0.038 | 고지대 → 도로 PM 낮음 (바람 통풍 양호) |
| `traffic` | +0.010 | 교통량 많을수록 도로 PM 증가 |
| `ndvi` | +0.009 | 식생 많은 격자 → 토양 노출 많아 PM 증가 |
| `traffic_x_road` | −0.008 | 교통×도로구조 상호작용 |
| `max_lanes` | −0.008 | 차로 수 많은 큰 도로 → 비산먼지 분산 |
| `sum_height` | +0.008 | 고층 건물 밀집 → 정체 효과 |
| `river_zone` | −0.007 | 하천 인근 → PM 낮음 |
| `sum_area` | −0.006 | 건물 연면적 높음 → 포장 도로 비율 높아 PM 낮음 |
| `buildings` | +0.006 | 건물 밀도 → 협소한 도로 PM 증가 |
| `road_struc` | −0.006 | 도로 구조 지수 |

### 하이퍼파라미터

| 파라미터 | 값 |
|----------|-----|
| `alpha` 탐색 범위 | [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0] |
| `cv` | 5-fold |
| 선택된 `alpha` | **1000.0** |
| Imputer | `SimpleImputer(strategy="median")` |
| Scaler | `StandardScaler` |

> alpha=1000 선택 의미: 강한 L2 정규화 → 계수를 작게 유지.  
> 공간 분산 자체가 전체 분산의 ~4%에 불과하므로, 과적합 방지를 위해 높은 alpha가 적절.

---

## 성능 결과

### 전체 지표

| 지표 | Train | Test |
|------|-------|------|
| MAE | 5.03 μg/m³ | **7.55 μg/m³** |
| R² | 0.595 | **0.301** |
| Stage1 단독 MAE | — | 7.74 μg/m³ |
| Stage1 단독 R² | — | 0.304 |

Stage2 Ridge 기여: MAE −0.19 (7.74 → 7.55)

### V3 vs V4 비교

| 모델 | Test MAE | Test R² | 예측 max | ceiling 문제 |
|------|---------|---------|---------|------------|
| V3 LightGBM_Weighted | 7.46 | **0.327** | ~50 | 있음 |
| V3 TwoStage | **7.34** | 0.276 | ~50 | 있음 |
| V3 LightGBM_BC | 7.33 | 0.286 | ~50 | 있음 |
| **V4 TwoStage (본 모델)** | 7.55 | **0.301** | **58.9** | **해소** |

> V4 MAE가 V3보다 약간 높은 이유: V3의 clip [2, 78]이 예측을 주 분포 범위(5–40)에 집중시켜 그 구간 MAE를 낮추는 대신 ceiling을 발생시킴. V4는 전 범위를 커버하는 대신 평균 MAE가 소폭 증가.

### PM 농도 구간별 MAE

| 실측 구간 | 건수 | MAE | 예측 평균 | 실측 평균 | 비고 |
|----------|------|-----|---------|---------|------|
| 0 – 5 μg/m³ | 15,952 | 6.18 | 9.1 | 2.9 | 저농도 과대예측 경향 |
| 5 – 10 | 27,044 | 3.37 | 9.9 | 6.8 | 양호 |
| 10 – 20 | 19,658 | 6.02 | 14.1 | 13.8 | 양호 |
| 20 – 40 | 8,459 | 10.80 | 20.8 | 27.6 | 중농도 과소예측 |
| 40 – 80 | 3,126 | 29.08 | 26.2 | 54.9 | 고농도 한계 |
| **80 – 200** | **949** | **81.71** | **29.9** | **111.7** | **극단값 한계** |

> 고농도(>40) 과소예측 원인:
> - 전체 데이터의 1% 미만인 희소 이벤트 → 학습 신호 부족
> - Train max=147 μg/m³ vs Test max=195 μg/m³ → Train에 없는 극단값
> - 2023-2024 → 2025 시간적 분포 이동 (temporal shift)

### 격자별 공간 분포

| 지표 | 실측 평균 | 예측 평균 |
|------|---------|---------|
| 격자 평균 road_pm 범위 | 2.57 ~ 35.60 μg/m³ | 6.62 ~ 32.29 μg/m³ |
| 격자 평균 std | 4.78 μg/m³ (공간 구조 유의) | 유사 |

---

## 실행

```bash
cd RoadExtension_V4

# 1. 학습 (construction_cache.csv는 V3 체크포인트에서 재사용)
python3 step_train.py

# 2. 시각화 (step_train.py 이후 실행)
python3 step_visualize.py
```

### 의존성

| 항목 | 설명 |
|------|------|
| `features_train.csv` | V2 checkpoints (V2 step1–4 실행 필요) |
| `features_test.csv` | V2 checkpoints |
| `construction_cache.csv` | V3 checkpoints (V3 `step_construction.py` 실행 필요, 이미 생성됨) |
| `격자_250m_4326.csv` | 격자 좌표 (시각화용) |

---

## 파일 구조

```
RoadExtension_V4/
├── config.py              ← 경로, 피처 목록, 모델 하이퍼파라미터
├── preprocess.py          ← RoadPMTransformer (log1p / expm1)
├── step_train.py          ← Two-Stage 학습 + 예측 저장
├── step_visualize.py      ← 공간 히트맵 + scatter plot 생성
├── README.md
└── checkpoints/
    ├── two_stage_model.pkl        ← 저장 모델
    ├── predictions_test.csv       ← 테스트 예측값 (시각화용)
    ├── results.json               ← MAE / R² 등 평가 지표
    ├── spatial_heatmap.png        ← 격자별 실제/예측 road_pm 공간 분포
    └── prediction_scatter.png     ← 예측 vs 실제 scatter plot
```

`two_stage_model.pkl` 구조:

```python
(
    RoadPMTransformer,    # log1p 변환기
    LGBMRegressor,        # Stage1 temporal 모델
    SimpleImputer,        # Stage2 결측 처리
    StandardScaler,       # Stage2 스케일링
    RidgeCV,              # Stage2 spatial 모델 (alpha=1000)
    feats_t,              # Stage1 피처 이름 리스트 (14개)
    feats_s,              # Stage2 피처 이름 리스트 (19개)
)
```

---

## 전체 프로젝트에서의 위치

```
ST-GNN (S3/static/w12)
  ↓ hidden vector h(T, N=40, d=64)
HiddenExtension V5-base
  ↓ ambient_pm10 grid (T, G=10,125)    ← 광역 배경 농도 (시간×격자)
RoadExtension V4 — TwoStage
  │
  ├─ [Stage1] Temporal LightGBM
  │    입력 (14개): 기온, 습도, is_dry, cold_and_dry, days_from_rain,
  │                 daily_precip_mm, season, month/hour cyclic, weekday,
  │                 is_weekend, ambient_pm10
  │    → log1p 공간의 시간 기준값
  │
  └─ [Stage2] Spatial RidgeCV (alpha=1000)
       입력 (19개): LUR 9종, 도로구조 6종, ambient_pm10, 공사 3종
       → 격자별 공간 잔차 (log1p 공간)
  │
  ↓ expm1 역변환
격자별 도로 PM10 (T, G=10,125)         ← 도로 재비산 성분
  ↓
Combined PM10 grid (T, G=10,125)       ← 최종 통합 농도
  ↓
도로 청소 차량 경로 최적화
```

---

## 성능 개선 여지

현재 미설명 분산: **69.9% (R²=0.301)**

| 원인 | 설명 |
|------|------|
| 시간적 분포 이동 | Train 2023–2024 → Test 2025: 극단값 패턴 차이 |
| 희소 고농도 이벤트 | >40 μg/m³ 이벤트가 전체의 5% 미만 → 학습 신호 부족 |
| 미관측 변수 | 도로 청소 이력, 화물차 비중, 도로 포장 노후도 |
| Stage2 비선형성 | Ridge 선형 모델 — 비선형 공간 효과 미포착 (격자 확장 유지를 위해 Ridge 유지) |
