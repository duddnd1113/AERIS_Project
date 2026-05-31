"""
[V4] Two-Stage 모델 학습
  log1p 변환 + 공사 피처 추가
  Stage 1: Temporal LightGBM (날씨/시간 → log1p 예측)
  Stage 2: Spatial Ridge     (LUR + ambient + 공사 → 잔차 보정)

실행:
    python3 step_train.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
import lightgbm as lgb

from config import (
    FEATURES_SRC_TR, FEATURES_SRC_TE, CONSTRUCTION_CACHE,
    CKPT_DIR, RESULTS_JSON, PRED_CSV, MODEL_PKL,
    TEMPORAL_STAGE_FEATS, SPATIAL_STAGE_FEATS,
    LGBM_STAGE1, RIDGE_ALPHAS, SAMPLE_WEIGHT_POWER,
)
from preprocess import RoadPMTransformer


def load_features(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def merge_construction(df: pd.DataFrame, cache: pd.DataFrame) -> pd.DataFrame:
    """공사 피처 left-join: (CELL_ID, date) 기준, 없으면 0 채움."""
    cache = cache.copy()
    cache["date"] = pd.to_datetime(cache["date"]).dt.normalize()

    df = df.copy()
    df["_date_key"] = df["date"].dt.normalize()

    merged = df.merge(
        cache[["CELL_ID", "date", "n_active_const", "total_amount_억", "is_large_const"]],
        left_on=["CELL_ID", "_date_key"],
        right_on=["CELL_ID", "date"],
        how="left",
        suffixes=("", "_const"),
    )
    merged.drop(columns=["_date_key", "date_const"], errors="ignore", inplace=True)
    for col in ["n_active_const", "total_amount_억", "is_large_const"]:
        merged[col] = merged[col].fillna(0.0)
    return merged


def resolve_feats(names: list, df: pd.DataFrame) -> list:
    return [f for f in names if f in df.columns and df[f].isna().mean() < 0.5]


def make_weights(y_raw: np.ndarray) -> np.ndarray:
    med = float(np.median(y_raw[y_raw > 0]))
    return (1.0 + (y_raw / med) ** SAMPLE_WEIGHT_POWER).astype(np.float32)


def build_stage2_features(df: pd.DataFrame, base_feats: list) -> pd.DataFrame:
    """
    Stage 2 용 피처 매트릭스 구성.
    base 공간 피처 + log 변환 + ambient×공간 상호작용 항.
    train/test 모두 동일한 함수로 호출 → 컬럼 일치 보장.
    """
    out = {}

    # ── 기본 공간 피처 ────────────────────────────────────────────────────
    for col in base_feats:
        out[col] = df[col].fillna(0).values if col in df.columns else np.zeros(len(df))

    # ── log 변환: 오른쪽 꼬리가 긴 피처 ──────────────────────────────────
    for col in ["n_active_const", "total_amount_억", "traffic"]:
        if col in df.columns:
            out[f"log1p_{col}"] = np.log1p(df[col].fillna(0).values)

    # ── ambient_pm10 기반 상호작용 항 ─────────────────────────────────────
    # 원리: ambient(광역 오염 수준) × 공간 특성 = 그 격자에서 road PM으로
    #       얼마나 증폭되는지 → Ridge가 학습 가능한 선형 신호
    if "ambient_pm10" in df.columns:
        amb = df["ambient_pm10"].values.astype(float)
        amb_med = float(np.nanmedian(amb[amb > 0])) if (amb > 0).any() else 1.0
        amb = np.where(np.isnan(amb), amb_med, amb)

        for col in ["traffic", "road_struc", "max_lanes",
                    "n_active_const", "highway_rank", "total_road_length_m"]:
            if col in df.columns:
                out[f"amb_x_{col}"] = amb * df[col].fillna(0).values

        # ambient 이차항 (비선형 ambient 효과를 선형화)
        out["ambient_sq"] = amb ** 2

        # 건조 조건 × ambient: 먼지 재비산 가중
        if "is_dry" in df.columns:
            out["amb_x_dry"] = amb * df["is_dry"].fillna(0).values

        # log1p(ambient) × traffic: 상대적 교통 오염 기여
        if "traffic" in df.columns:
            out["log_amb_x_traffic"] = np.log1p(amb) * df["traffic"].fillna(0).values

    return pd.DataFrame(out, index=df.index)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray):
    y_pred = np.clip(y_pred, 0, None)
    return float(mean_absolute_error(y_true, y_pred)), float(r2_score(y_true, y_pred))


def train_two_stage(df_tr, df_te):
    tr = RoadPMTransformer()
    y_tr_raw = df_tr["road_pm"].values
    y_te_raw = df_te["road_pm"].values
    y_tr_log = tr.fit_transform(y_tr_raw)
    y_te_log = tr.transform(y_te_raw)

    feats_t = resolve_feats(TEMPORAL_STAGE_FEATS, df_tr)
    feats_t = [f for f in feats_t if f in df_te.columns]
    feats_s = resolve_feats(SPATIAL_STAGE_FEATS, df_tr)
    feats_s = [f for f in feats_s if f in df_te.columns]

    print(f"  Stage1 피처 {len(feats_t)}개: {feats_t}")
    print(f"  Stage2 피처 {len(feats_s)}개: {feats_s}")

    X_tr_t = df_tr[feats_t].values
    X_te_t = df_te[feats_t].values
    X_tr_s = df_tr[feats_s].values
    X_te_s = df_te[feats_s].values

    # ── Stage 1: Temporal LightGBM ──────────────────────────────────────────
    print("\n  [Stage 1] Temporal LightGBM 학습...")
    weights = make_weights(y_tr_raw)
    w1 = weights if SAMPLE_WEIGHT_POWER > 0 else None
    m1 = lgb.LGBMRegressor(**LGBM_STAGE1)
    m1.fit(
        X_tr_t, y_tr_log,
        sample_weight=w1,
        eval_set=[(X_te_t, y_te_log)],
        callbacks=[lgb.early_stopping(100, verbose=False),
                   lgb.log_evaluation(period=100)],
    )
    stage1_tr = m1.predict(X_tr_t)
    stage1_te = m1.predict(X_te_t)
    resid_tr  = y_tr_log - stage1_tr

    s1_te_mae, s1_te_r2 = evaluate(y_te_raw, tr.inverse_transform(stage1_te))
    print(f"  Stage1 단독  MAE={s1_te_mae:.4f}  R²={s1_te_r2:.4f}")
    print(f"  잔차 std: {resid_tr.std():.4f}  (전체 log std: {y_tr_log.std():.4f})")

    # ── Stage 2: Spatial RidgeCV ────────────────────────────────────────────
    print("\n  [Stage 2] Spatial RidgeCV 학습...")
    imp = SimpleImputer(strategy="median")
    sc  = StandardScaler()
    X_tr_s2 = sc.fit_transform(imp.fit_transform(X_tr_s))
    X_te_s2 = sc.transform(imp.transform(X_te_s))

    m2 = RidgeCV(alphas=RIDGE_ALPHAS, cv=5)
    m2.fit(X_tr_s2, resid_tr)
    print(f"  RidgeCV 선택 alpha: {m2.alpha_:.4f}")

    resid_te_pred = m2.predict(X_te_s2)
    resid_tr_pred = m2.predict(X_tr_s2)

    # ── 최종 예측 ─────────────────────────────────────────────────────────
    pred_te = tr.inverse_transform(stage1_te + resid_te_pred)
    pred_tr = tr.inverse_transform(stage1_tr + resid_tr_pred)

    tr_mae, tr_r2 = evaluate(y_tr_raw, pred_tr)
    te_mae, te_r2 = evaluate(y_te_raw, pred_te)

    return (tr, m1, imp, sc, m2, feats_t, feats_s), \
           pred_tr, pred_te, (tr_mae, tr_r2), (te_mae, te_r2), y_te_raw


def main():
    print("=== [V4] Two-Stage Road PM 학습 ===\n")

    # 데이터 로드
    print("[1] 데이터 로드...")
    df_tr = load_features(FEATURES_SRC_TR)
    df_te = load_features(FEATURES_SRC_TE)
    print(f"  Train: {len(df_tr):,}건  Test: {len(df_te):,}건")

    # 공사 피처 병합
    print("\n[2] 공사 피처 병합...")
    cache = pd.read_csv(CONSTRUCTION_CACHE)
    df_tr = merge_construction(df_tr, cache)
    df_te = merge_construction(df_te, cache)
    const_cover_tr = (df_tr["n_active_const"] > 0).mean()
    const_cover_te = (df_te["n_active_const"] > 0).mean()
    print(f"  공사 데이터 커버율: Train={const_cover_tr*100:.1f}%  Test={const_cover_te*100:.1f}%")

    # 전처리 요약
    print("\n[3] 전처리")
    tr = RoadPMTransformer()
    tr.summary()
    y_all = df_tr["road_pm"]
    print(f"  road_pm — mean={y_all.mean():.2f}  median={y_all.median():.2f}  "
          f"max={y_all.max():.2f}  log1p_std={np.log1p(y_all).std():.4f}")

    # 학습
    print("\n[4] Two-Stage 학습")
    artifacts, pred_tr, pred_te, (tr_mae, tr_r2), (te_mae, te_r2), y_te_raw \
        = train_two_stage(df_tr, df_te)

    # 고농도 구간 성능
    mask_high = y_te_raw > 50
    if mask_high.sum() > 0:
        hi_mae, hi_r2 = evaluate(y_te_raw[mask_high], pred_te[mask_high])
        print(f"\n  [고농도 > 50] n={mask_high.sum()}, MAE={hi_mae:.4f}  R²={hi_r2:.4f}")

    print("\n" + "="*50)
    print(f"  Train MAE={tr_mae:.4f}  R²={tr_r2:.4f}")
    print(f"  Test  MAE={te_mae:.4f}  R²={te_r2:.4f}")
    print(f"  (V3 TwoStage 기준 MAE / R² 와 비교)")
    print("="*50)

    # 저장
    joblib.dump(artifacts, MODEL_PKL)
    print(f"\n모델 저장: {MODEL_PKL}")

    # 예측값 CSV 저장 (시각화용)
    pred_df = df_te[["CELL_ID", "date", "hour"]].copy()
    pred_df["road_pm_actual"] = y_te_raw
    pred_df["road_pm_pred"]   = pred_te
    pred_df.to_csv(PRED_CSV, index=False)
    print(f"예측 저장: {PRED_CSV}")

    # 결과 JSON
    results = {
        "model":      "TwoStage_log1p",
        "transform":  "log1p",
        "n_train":    int(len(df_tr)),
        "n_test":     int(len(df_te)),
        "train_mae":  tr_mae, "train_r2": tr_r2,
        "test_mae":   te_mae, "test_r2":  te_r2,
        "high_pm_mae": float(hi_mae) if mask_high.sum() > 0 else None,
        "high_pm_r2":  float(hi_r2)  if mask_high.sum() > 0 else None,
    }
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"결과 저장: {RESULTS_JSON}")


if __name__ == "__main__":
    main()
