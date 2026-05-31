"""
[V4] 특정 날짜 하루 — 서울 전역 10,125 격자 road PM10 예측 시각화 (Folium HTML)

모델을 불러 관측치가 없는 격자에도 예측을 확산한 뒤 실제 서울 지도 위에 표시.

레이어 3개가 하나의 HTML에 포함되며, 지도 좌상단 체크박스로 on/off:
  ① Predicted 전역   — 서울 전체 10,125 격자 예측값 (기본 ON)
  ② Predicted 관측도로 — 정답 있는 도로 셀만 예측값 (기본 OFF)
  ③ Actual 관측      — 정답 있는 도로 셀 실측값   (기본 OFF)

실행:
    python3 step_visualize_daily_map.py                   # 고농도 날 자동 선택
    python3 step_visualize_daily_map.py --date 2025-03-17 # 성능 최고 날 (R²=0.316)
    python3 step_visualize_daily_map.py --date 2025-03-17 --hour 12
"""
import os, sys, argparse, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import joblib
import folium
from sklearn.impute import SimpleImputer

from config import (
    PRED_CSV, GRID_CSV, CKPT_DIR, MODEL_PKL,
    FEATURES_SRC_TR, FEATURES_SRC_TE, CONSTRUCTION_CACHE,
    TEMPORAL_STAGE_FEATS, SPATIAL_STAGE_FEATS,
)

GRID_LUR_CSV = (
    "/home/data/youngwoong/ST-GNN_Dataset/Data_Preprocessed"
    "/Land Use Regression/격자 기본/격자_250m_4326_with_lur.csv"
)

SEOUL_CENTER = [37.5665, 126.9780]
HALF_LAT = 125 / 111320
HALF_LON = 125 / (111320 * math.cos(math.radians(37.5)))


# ── 인수 파싱 ─────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--date", default=None, help="날짜 YYYY-MM-DD (미지정시 고농도 날 자동)")
    p.add_argument("--hour", type=int, default=None, help="시간 9~16 (미지정시 하루 평균)")
    p.add_argument("--out",  default=None, help="출력 HTML 경로")
    return p.parse_args()


# ── 모델 로드 ─────────────────────────────────────────────────────────────────
def load_model():
    tr, m1, imp, sc, m2, feats_t, feats_s = joblib.load(MODEL_PKL)
    return tr, m1, imp, sc, m2, feats_t, feats_s


# ── 공간 피처 저장소 구축 (전체 10,125 격자) ──────────────────────────────────
def build_spatial_store(grid: pd.DataFrame) -> pd.DataFrame:
    """
    전체 격자의 공간 피처 DataFrame 반환 (10,125행).
    - 관측 격자: features_train/test 에서 실제 값 사용
    - 미관측 격자: LUR 기본 피처 + 도로 피처 0 처리
    """
    # 관측 격자 공간 피처 (train + test 합쳐서 CELL_ID별 평균)
    spatial_cols = [c for c in SPATIAL_STAGE_FEATS
                    if c not in ("ambient_pm10", "n_active_const",
                                 "total_amount_억", "is_large_const")]
    tr = pd.read_csv(FEATURES_SRC_TR)
    te = pd.read_csv(FEATURES_SRC_TE)
    obs = pd.concat([tr, te], ignore_index=True)
    avail = [c for c in spatial_cols if c in obs.columns]
    cell_feats = obs.groupby("CELL_ID")[avail].mean().reset_index()

    # LUR 기본 피처 (전체 격자)
    lur = pd.read_csv(GRID_LUR_CSV)
    lur = lur.rename(columns={"NDVI": "ndvi", "IBI": "ibi"})

    # 전체 격자 기반 DataFrame 구성
    base = grid[["CELL_ID", "lat", "lon"]].merge(
        lur[["CELL_ID", "ndvi", "ibi", "buildings", "greenspace",
             "road_struc", "river_zone"]],
        on="CELL_ID", how="left"
    )

    # 관측 격자 피처로 덮어쓰기
    base = base.merge(cell_feats.add_suffix("_obs").rename(
        columns={"CELL_ID_obs": "CELL_ID"}),
        on="CELL_ID", how="left"
    )
    for col in avail:
        obs_col = f"{col}_obs"
        if obs_col in base.columns:
            # 관측 피처 우선, 없으면 LUR 값 유지 (lur에 있는 피처는 이미 base에 있음)
            mask = base[obs_col].notna()
            if col in base.columns:
                base.loc[mask, col] = base.loc[mask, obs_col]
            else:
                base[col] = base[obs_col]
    # obs suffix 컬럼 제거
    drop_cols = [c for c in base.columns if c.endswith("_obs")]
    base.drop(columns=drop_cols, inplace=True)

    return base


# ── 날짜/시간 temporal 피처 추출 ─────────────────────────────────────────────
def get_temporal_row(date_str: str, hour, feat_df: pd.DataFrame) -> dict:
    """
    선택 날짜/시간의 temporal 피처를 관측 데이터에서 추출 (전 격자 공통 적용).
    관측 없는 날짜면 date-math 피처만 계산하고 weather는 NaN 반환.
    """
    ts = pd.Timestamp(date_str)
    feat_df = feat_df.copy()
    feat_df["date"] = pd.to_datetime(feat_df["date"])
    day = feat_df[feat_df["date"].dt.date == ts.date()]
    if hour is not None:
        day = day[day["hour"] == hour]

    weather_cols = ["기온", "습도", "is_dry", "cold_and_dry",
                    "days_from_rain", "daily_precip_mm", "ambient_pm10"]

    if day.empty:
        row = {}
    else:
        row = {c: float(day[c].mean()) for c in weather_cols if c in day.columns}

    # 날짜 파생 피처 (항상 계산)
    eff_hour = hour if hour is not None else 12
    row["month_sin"]  = math.sin(2 * math.pi * ts.month / 12)
    row["month_cos"]  = math.cos(2 * math.pi * ts.month / 12)
    row["hour_sin"]   = math.sin(2 * math.pi * eff_hour / 24)
    row["hour_cos"]   = math.cos(2 * math.pi * eff_hour / 24)
    row["weekday"]    = float(ts.weekday())
    row["is_weekend"] = float(ts.weekday() >= 5)
    row["season"]     = float([0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 0][ts.month - 1])
    return row


# ── 전체 격자 피처 매트릭스 구성 ──────────────────────────────────────────────
def build_full_feature_matrix(
    spatial_store: pd.DataFrame,
    temporal_row: dict,
    construction_cache: pd.DataFrame,
    date_str: str,
    feats_t: list,
    feats_s: list,
) -> pd.DataFrame:
    df = spatial_store.copy()

    # temporal 피처 브로드캐스트
    for col, val in temporal_row.items():
        df[col] = val

    # 공사 피처 join
    cache = construction_cache.copy()
    cache["date"] = pd.to_datetime(cache["date"]).dt.normalize()
    target_date = pd.Timestamp(date_str).normalize()
    day_const = cache[cache["date"] == target_date][
        ["CELL_ID", "n_active_const", "total_amount_억", "is_large_const"]]
    if not day_const.empty:
        df = df.merge(day_const, on="CELL_ID", how="left")
    for col in ["n_active_const", "total_amount_억", "is_large_const"]:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = df[col].fillna(0.0)

    return df


# ── 추론 ──────────────────────────────────────────────────────────────────────
def infer_all_cells(df_full, m1, imp, sc, m2, tr_model, feats_t, feats_s):
    # Stage 1 (temporal)
    avail_t = [f for f in feats_t if f in df_full.columns]
    X_t = df_full[avail_t].fillna(0).values
    stage1_log = m1.predict(X_t)

    # Stage 2 (spatial residual)
    avail_s = [f for f in feats_s if f in df_full.columns]
    X_s_raw = df_full[avail_s].values
    X_s = sc.transform(imp.transform(X_s_raw))
    resid_pred = m2.predict(X_s)

    pred_log = stage1_log + resid_pred
    pred_pm  = tr_model.inverse_transform(pred_log)
    return pred_pm


# ── 색상 변환 ─────────────────────────────────────────────────────────────────
def values_to_hex(values, cmap_name, vmin, vmax):
    cmap = plt.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    out = []
    for v in values:
        rgba = cmap(norm(float(np.clip(v, vmin, vmax))))
        out.append("#{:02x}{:02x}{:02x}".format(
            int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255)))
    return out


# ── Folium 지도 구성 ──────────────────────────────────────────────────────────
def add_layer(m, rows, label, show):
    fg = folium.FeatureGroup(name=label, show=show)
    for r in rows:
        lat, lon = r["lat"], r["lon"]
        corners = [
            [lat - HALF_LAT, lon - HALF_LON],
            [lat - HALF_LAT, lon + HALF_LON],
            [lat + HALF_LAT, lon + HALF_LON],
            [lat + HALF_LAT, lon - HALF_LON],
        ]
        folium.Polygon(
            locations=corners,
            color="none",
            fill=True,
            fill_color=r["color"],
            fill_opacity=0.75,
            tooltip=folium.Tooltip(r["tooltip"]),
        ).add_to(fg)
    fg.add_to(m)


def colorbar_html(vmin, vmax, cmap_name="YlOrRd"):
    cmap = plt.get_cmap(cmap_name)
    stops = []
    for i in range(11):
        t = i / 10
        rgba = cmap(t)
        stops.append("#{:02x}{:02x}{:02x} {:d}%".format(
            int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255), int(t*100)))
    gradient = ", ".join(stops)
    ticks = [vmin + (vmax - vmin) * t for t in [0, 0.25, 0.5, 0.75, 1.0]]
    tick_html = "".join(
        f'<span style="position:absolute;left:{i*25}%;'
        f'transform:translateX(-50%);font-size:10px;color:#333">{v:.0f}</span>'
        for i, v in enumerate(ticks))
    return f"""
    <div style="position:fixed;bottom:40px;right:20px;z-index:1000;
                background:white;padding:10px 14px 28px 14px;
                border:1px solid #aaa;border-radius:6px;
                box-shadow:2px 2px 6px rgba(0,0,0,.3);font-family:Arial">
      <div style="font-size:12px;font-weight:bold;margin-bottom:6px;color:#222">
        road PM10 (μg/m³)</div>
      <div style="position:relative;width:200px;height:14px;
                  background:linear-gradient(to right,{gradient});
                  border-radius:3px;border:1px solid #ccc"></div>
      <div style="position:relative;width:200px;height:18px;margin-top:2px">{tick_html}</div>
    </div>"""


def build_map(df_pred, df_obs, date_str, hour, out_path):
    hour_str = f"{hour:02d}:00" if hour is not None else "하루 평균"

    # 공통 컬러 범위 (전역예측 + 실측 모두 포함)
    all_vals = df_pred["pred_pm"].values
    if df_obs is not None and not df_obs.empty:
        all_vals = np.concatenate([all_vals, df_obs["actual"].values])
    vmax = float(np.nanpercentile(all_vals, 98))
    vmax = max(vmax, 10.0)
    vmin = 0.0

    n_pred = len(df_pred)
    n_obs  = len(df_obs) if df_obs is not None else 0

    m = folium.Map(location=SEOUL_CENTER, zoom_start=11, tiles="CartoDB positron")
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)
    folium.TileLayer("CartoDB dark_matter", name="Dark").add_to(m)

    # ── ① Predicted 전역: 서울 전체 10,125셀 (기본 ON) ─────────────────────
    rows = []
    for _, r in df_pred.iterrows():
        col = values_to_hex([r["pred_pm"]], "YlOrRd", vmin, vmax)[0]
        rows.append({
            "lat": r["lat"], "lon": r["lon"], "color": col,
            "tooltip": (
                f"<b>{r['CELL_ID']}</b><br>"
                f"Predicted (전역): <b>{r['pred_pm']:.1f} μg/m³</b>"
            )
        })
    add_layer(m, rows, f"① Predicted 전역 ({n_pred:,}셀)", show=True)

    if df_obs is not None and not df_obs.empty:
        # ── ② Predicted 관측도로: 정답 있는 셀만 예측값 (기본 OFF) ───────────
        rows = []
        for _, r in df_obs.iterrows():
            col = values_to_hex([r["pred_pm"]], "YlOrRd", vmin, vmax)[0]
            rows.append({
                "lat": r["lat"], "lon": r["lon"], "color": col,
                "tooltip": (
                    f"<b>{r['CELL_ID']}</b><br>"
                    f"Predicted: <b>{r['pred_pm']:.1f} μg/m³</b><br>"
                    f"Actual: {r['actual']:.1f} μg/m³<br>"
                    f"Bias: {r['pred_pm']-r['actual']:+.1f} μg/m³"
                )
            })
        add_layer(m, rows, f"② Predicted 관측도로 ({n_obs:,}셀)", show=False)

        # ── ③ Actual 관측: 실측값 (기본 OFF) ─────────────────────────────────
        rows = []
        mae = float(np.mean(np.abs(df_obs["pred_pm"].values - df_obs["actual"].values)))
        r2_num = np.sum((df_obs["pred_pm"].values - df_obs["actual"].values) ** 2)
        r2_den = np.sum((df_obs["actual"].values - df_obs["actual"].values.mean()) ** 2)
        r2 = float(1 - r2_num / r2_den) if r2_den > 0 else float("nan")
        for _, r in df_obs.iterrows():
            col = values_to_hex([r["actual"]], "YlOrRd", vmin, vmax)[0]
            rows.append({
                "lat": r["lat"], "lon": r["lon"], "color": col,
                "tooltip": (
                    f"<b>{r['CELL_ID']}</b><br>"
                    f"Actual: <b>{r['actual']:.1f} μg/m³</b><br>"
                    f"Predicted: {r['pred_pm']:.1f} μg/m³<br>"
                    f"Bias: {r['pred_pm']-r['actual']:+.1f} μg/m³"
                )
            })
        add_layer(m, rows, f"③ Actual 관측 ({n_obs:,}셀)", show=False)
    else:
        mae, r2 = float("nan"), float("nan")

    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(folium.Element(colorbar_html(vmin, vmax)))

    pv = df_pred["pred_pm"].values
    title_html = f"""
    <div style="position:fixed;top:12px;left:50%;transform:translateX(-50%);
                z-index:1000;background:white;padding:8px 16px;
                border:1px solid #aaa;border-radius:6px;
                box-shadow:2px 2px 6px rgba(0,0,0,.3);
                font-family:Arial;font-size:13px;font-weight:bold;color:#222">
      서울 Road PM10 &nbsp;|&nbsp; {date_str} &nbsp;{hour_str}
      &nbsp;&nbsp;<span style="font-weight:normal;font-size:11px;color:#555">
        pred_mean={pv.mean():.1f} &nbsp; pred_max={pv.max():.1f} μg/m³ &nbsp;|&nbsp;
        MAE={mae:.2f} &nbsp; R²={r2:.3f}
      </span>
    </div>"""
    m.get_root().html.add_child(folium.Element(title_html))

    m.save(out_path)
    print(f"  저장: {out_path}")


# ── 메인 ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    print("=== [V4] 서울 전역 road PM10 지도 시각화 ===\n")

    # 데이터 로드
    print("[1] 데이터 로드...")
    pred_obs = pd.read_csv(PRED_CSV, parse_dates=["date"])
    grid     = pd.read_csv(GRID_CSV)[["CELL_ID", "lat", "lon"]]
    tr_model, m1, imp, sc, m2, feats_t, feats_s = load_model()
    print(f"  grid: {len(grid):,}셀  model: {MODEL_PKL}")

    # 날짜 선택
    date_str = args.date if args.date else str(
        pred_obs.groupby(pred_obs["date"].dt.date)["road_pm_actual"].mean().idxmax())
    print(f"  날짜: {date_str}  시간: {args.hour if args.hour is not None else '전체(평균)'}")

    # 공간 피처 저장소 (전체 격자)
    print("\n[2] 전체 격자 공간 피처 구성...")
    spatial_store = build_spatial_store(grid)
    print(f"  격자 수: {len(spatial_store):,}")

    # Temporal 피처 (관측 데이터에서 추출)
    print("\n[3] Temporal 피처 추출...")
    feat_all = pd.concat([
        pd.read_csv(FEATURES_SRC_TR),
        pd.read_csv(FEATURES_SRC_TE)
    ], ignore_index=True)
    temporal_row = get_temporal_row(date_str, args.hour, feat_all)
    print(f"  ambient_pm10={temporal_row.get('ambient_pm10', 'N/A'):.1f}  "
          f"기온={temporal_row.get('기온', float('nan')):.1f}  "
          f"습도={temporal_row.get('습도', float('nan')):.1f}")

    # 공사 피처
    construction_cache = pd.read_csv(CONSTRUCTION_CACHE)

    # 전체 피처 매트릭스
    print("\n[4] 전체 격자 피처 매트릭스 구성...")
    df_full = build_full_feature_matrix(
        spatial_store, temporal_row, construction_cache,
        date_str, feats_t, feats_s)
    print(f"  피처 매트릭스: {df_full.shape}")

    # 추론
    print("\n[5] 전체 격자 추론 (Stage1 + Stage2)...")
    pred_pm = infer_all_cells(df_full, m1, imp, sc, m2, tr_model, feats_t, feats_s)
    df_full["pred_pm"] = pred_pm
    print(f"  pred_mean={pred_pm.mean():.2f}  pred_max={pred_pm.max():.2f}  "
          f"pred_min={pred_pm.min():.2f}")

    # 관측 격자 실제값 (비교용)
    day_obs = pred_obs[pred_obs["date"].dt.date == pd.Timestamp(date_str).date()].copy()
    if args.hour is not None:
        day_obs = day_obs[day_obs["hour"] == args.hour]
    if not day_obs.empty:
        obs_stats = (day_obs.groupby("CELL_ID")
                     .agg(actual=("road_pm_actual", "mean"),
                          pred_pm=("road_pm_pred", "mean"))
                     .reset_index()
                     .merge(grid, on="CELL_ID", how="inner"))
        print(f"\n  관측 격자: {len(obs_stats):,}셀  "
              f"actual_mean={obs_stats['actual'].mean():.2f}")
    else:
        obs_stats = None

    # 지도 생성
    hour_tag = f"_h{args.hour:02d}" if args.hour is not None else "_daily"
    out_path = args.out or os.path.join(
        CKPT_DIR, f"map_{date_str.replace('-','')}{hour_tag}.html")

    print(f"\n[6] 지도 생성 (레이어 3종 포함)...")
    build_map(df_full, obs_stats, date_str, args.hour, out_path)

    print("\n완료. 브라우저에서 HTML 파일을 열어 확인하세요.")


if __name__ == "__main__":
    main()
