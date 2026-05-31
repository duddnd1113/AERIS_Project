"""
RoadExtension V4 — 설정
- Two-Stage 단일 모델
- log1p 변환 (clip 없음, 극단값 학습 가능)
- 공사 피처 추가 (Stage 2 spatial)
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, "..")

# ── 입력 데이터 ────────────────────────────────────────────────────────────
V2_DIR          = os.path.join(ROOT_DIR, "RoadExtension_V2")
FEATURES_SRC_TR = os.path.join(V2_DIR, "checkpoints", "features_train.csv")
FEATURES_SRC_TE = os.path.join(V2_DIR, "checkpoints", "features_test.csv")

# V3에서 생성한 공사 피처 캐시 재사용
CONSTRUCTION_CACHE = os.path.join(
    ROOT_DIR, "RoadExtension_V3", "checkpoints", "construction_cache.csv"
)

# 격자 좌표 (시각화용)
GRID_CSV = (
    "/home/data/youngwoong/ST-GNN_Dataset/Data_Preprocessed"
    "/Land Use Regression/격자 기본/격자_250m_4326.csv"
)

# ── V4 저장 경로 ───────────────────────────────────────────────────────────
CKPT_DIR       = os.path.join(BASE_DIR, "checkpoints")
RESULTS_JSON   = os.path.join(CKPT_DIR, "results.json")
PRED_CSV       = os.path.join(CKPT_DIR, "predictions_test.csv")
MODEL_PKL      = os.path.join(CKPT_DIR, "two_stage_model.pkl")
HEATMAP_PNG    = os.path.join(CKPT_DIR, "spatial_heatmap.png")
SCATTER_PNG    = os.path.join(CKPT_DIR, "prediction_scatter.png")
os.makedirs(CKPT_DIR, exist_ok=True)

# ── 피처 구성 ──────────────────────────────────────────────────────────────
# Stage 1: 시간/날씨 + ambient (temporal component)
# ambient_pm10: 당일 광역 오염 수준 — 가장 강한 temporal 예측 변수
TEMPORAL_STAGE_FEATS = [
    "month_sin", "month_cos", "hour_sin", "hour_cos",
    "weekday", "is_weekend", "season",
    "기온", "습도", "is_dry", "cold_and_dry",
    "days_from_rain", "daily_precip_mm",
    "ambient_pm10",
]

# Stage 2: 공간/도로/공사 (spatial residual)
SPATIAL_STAGE_FEATS = [
    # LUR 피처
    "buildings", "greenspace", "road_struc", "river_zone",
    "ndvi", "ibi", "elev_mean", "sum_area", "sum_height",
    # 도로 구조
    "traffic", "highway_rank", "max_lanes", "mean_gvi",
    "total_road_length_m", "traffic_x_road",
    # ambient
    "ambient_pm10",
    # 공사 피처 (V3 construction_cache)
    "n_active_const", "total_amount_억", "is_large_const",
]

# ── 모델 하이퍼파라미터 ────────────────────────────────────────────────────
LGBM_STAGE1 = dict(
    n_estimators=1000, learning_rate=0.03, max_depth=8,
    num_leaves=63, min_child_samples=10, subsample=0.8,
    colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
    n_jobs=-1, random_state=42, verbose=-1,
)

RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]  # RidgeCV 탐색 범위

# 고농도 샘플 가중치: weight = 1 + (pm / median)^POWER
# 1.0 → y=50 시 6배, y=100 시 11배, y=150 시 16배 가중
SAMPLE_WEIGHT_POWER = 0.5   # 0.0=no weight (best MAE), 0.5=best R²
