"""
[V4] 특정 날짜 하루 공간 분포 시각화

실행 예시:
    python3 step_visualize_daily.py                        # 고농도 일 자동 선택
    python3 step_visualize_daily.py --date 2025-01-17      # 날짜 지정
    python3 step_visualize_daily.py --date 2025-01-17 --hour 9   # 특정 시간
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

import matplotlib.font_manager as fm
_korean_fonts = [f.name for f in fm.fontManager.ttflist
                 if any(k in f.name for k in ("Gothic", "Nanum", "Malgun", "AppleGothic"))]
plt.rcParams["font.family"] = _korean_fonts[0] if _korean_fonts else "DejaVu Sans"

from config import PRED_CSV, GRID_CSV, CKPT_DIR


SEOUL_LON = (126.75, 127.18)
SEOUL_LAT = (37.43, 37.70)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=None,
                        help="시각화 날짜 (YYYY-MM-DD). 미지정시 최고농도 날 자동 선택")
    parser.add_argument("--hour", type=int, default=None,
                        help="특정 시간 (9~16). 미지정시 하루 평균")
    parser.add_argument("--out", type=str, default=None,
                        help="출력 파일 경로. 미지정시 checkpoints/daily_<date>[_h<hour>].png")
    return parser.parse_args()


def load_data():
    pred = pd.read_csv(PRED_CSV, parse_dates=["date"])
    grid = pd.read_csv(GRID_CSV)[["CELL_ID", "lat", "lon"]]
    return pred, grid


def select_date(pred: pd.DataFrame) -> str:
    """고농도 평균이 가장 높은 날 자동 선택."""
    best = (pred.groupby(pred["date"].dt.date)["road_pm_actual"]
            .mean().idxmax())
    return str(best)


def filter_day(pred: pd.DataFrame, date_str: str, hour) -> pd.DataFrame:
    day = pred[pred["date"].dt.date == pd.Timestamp(date_str).date()].copy()
    if hour is not None:
        day = day[day["hour"] == hour]
    return day


def compute_cell_stats(day: pd.DataFrame) -> pd.DataFrame:
    return (day.groupby("CELL_ID")
            .agg(actual=("road_pm_actual", "mean"),
                 pred=("road_pm_pred", "mean"),
                 n=("road_pm_actual", "count"))
            .reset_index())


def draw_scatter_map(ax, lon, lat, values, title, cmap, norm=None, vmin=None, vmax=None):
    kw = {"norm": norm} if norm is not None else {"vmin": vmin, "vmax": vmax}
    sc = ax.scatter(lon, lat, c=values, cmap=cmap, s=5, alpha=0.9,
                    rasterized=True, **kw)
    ax.set_xlim(*SEOUL_LON)
    ax.set_ylim(*SEOUL_LAT)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Longitude", fontsize=8)
    ax.set_ylabel("Latitude", fontsize=8)
    ax.tick_params(labelsize=7)
    cb = plt.colorbar(sc, ax=ax, shrink=0.85)
    cb.set_label("road_pm (μg/m³)", fontsize=8)
    return sc


def make_daily_figure(stats, grid, date_str, hour, out_path):
    merged = stats.merge(grid, on="CELL_ID", how="inner").dropna(
        subset=["lat", "lon", "actual", "pred"])

    if merged.empty:
        print("  [경고] 격자 좌표 병합 결과 없음")
        return

    lon = merged["lon"].values
    lat = merged["lat"].values
    actual = merged["actual"].values
    pred   = merged["pred"].values
    bias   = pred - actual

    n_cells = len(merged)
    mae  = float(np.mean(np.abs(bias)))
    rmse = float(np.sqrt(np.mean(bias**2)))
    r2   = float(1 - np.sum(bias**2) / np.sum((actual - actual.mean())**2)) \
           if actual.std() > 0 else float("nan")

    # 공통 컬러 범위
    vmax_map = float(np.percentile(np.concatenate([actual, pred]), 98))
    vmax_map = max(vmax_map, 10.0)
    vmin_map = 0.0
    norm = mcolors.Normalize(vmin=vmin_map, vmax=vmax_map)

    fig = plt.figure(figsize=(18, 12))
    gs  = GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.28)

    hour_str = f"Hour {hour:02d}:00" if hour is not None else "Daily mean"
    title_main = (f"V4 Road PM10 — {date_str}  ({hour_str})\n"
                  f"cells={n_cells:,}  MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.3f}")

    # 1. Actual
    ax1 = fig.add_subplot(gs[0, 0])
    draw_scatter_map(ax1, lon, lat, actual, f"Actual road_pm\n{date_str} {hour_str}",
                     "YlOrRd", norm=norm)

    # 2. Predicted
    ax2 = fig.add_subplot(gs[0, 1])
    draw_scatter_map(ax2, lon, lat, pred, f"Predicted road_pm\n{date_str} {hour_str}",
                     "YlOrRd", norm=norm)

    # 3. Bias (pred - actual)
    ax3 = fig.add_subplot(gs[0, 2])
    vlim = float(np.percentile(np.abs(bias), 95))
    sc3 = ax3.scatter(lon, lat, c=bias, cmap="RdBu_r",
                      s=5, alpha=0.9, vmin=-vlim, vmax=vlim, rasterized=True)
    ax3.set_xlim(*SEOUL_LON); ax3.set_ylim(*SEOUL_LAT)
    ax3.set_title(f"Bias (pred − actual)\nbias_mean={bias.mean():.2f}  std={bias.std():.2f}",
                  fontsize=11)
    ax3.set_xlabel("Longitude", fontsize=8); ax3.set_ylabel("Latitude", fontsize=8)
    ax3.tick_params(labelsize=7)
    cb3 = plt.colorbar(sc3, ax=ax3, shrink=0.85)
    cb3.set_label("Δ road_pm (μg/m³)", fontsize=8)

    # 4. Scatter plot: pred vs actual
    ax4 = fig.add_subplot(gs[1, 0])
    c4 = np.clip(actual, 0, vmax_map)
    sc4 = ax4.scatter(actual, pred, c=c4, cmap="YlOrRd",
                      alpha=0.5, s=6, rasterized=True,
                      vmin=vmin_map, vmax=vmax_map)
    lim = max(float(actual.max()), float(pred.max()), 10)
    ax4.plot([0, lim], [0, lim], "k--", lw=1.2)
    ax4.set_xlim(0, lim); ax4.set_ylim(0, lim)
    ax4.set_xlabel("Actual road_pm (μg/m³)", fontsize=9)
    ax4.set_ylabel("Predicted road_pm (μg/m³)", fontsize=9)
    ax4.set_title(f"Scatter (per-cell mean)\nMAE={mae:.2f}  R²={r2:.3f}", fontsize=11)
    plt.colorbar(sc4, ax=ax4, shrink=0.85, label="Actual road_pm")

    # 5. Actual distribution histogram
    ax5 = fig.add_subplot(gs[1, 1])
    bins = np.linspace(0, max(actual.max(), pred.max(), 20), 40)
    ax5.hist(actual, bins=bins, alpha=0.6, color="steelblue",  label="Actual", edgecolor="white")
    ax5.hist(pred,   bins=bins, alpha=0.6, color="orangered",  label="Predicted", edgecolor="white")
    ax5.axvline(actual.mean(), color="steelblue", linestyle="--", lw=1.2,
                label=f"Actual μ={actual.mean():.1f}")
    ax5.axvline(pred.mean(),   color="orangered",  linestyle="--", lw=1.2,
                label=f"Pred μ={pred.mean():.1f}")
    ax5.set_xlabel("road_pm (μg/m³)", fontsize=9)
    ax5.set_ylabel("Grid count", fontsize=9)
    ax5.set_title("Per-cell mean distribution", fontsize=11)
    ax5.legend(fontsize=8)

    # 6. Bias histogram
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.hist(bias, bins=40, color="mediumpurple", edgecolor="white", linewidth=0.3)
    ax6.axvline(0, color="red", linestyle="--", lw=1.2)
    ax6.axvline(bias.mean(), color="orange", linestyle="-", lw=1.2,
                label=f"mean={bias.mean():.2f}")
    ax6.set_xlabel("Pred − Actual road_pm (μg/m³)", fontsize=9)
    ax6.set_ylabel("Grid count", fontsize=9)
    ax6.set_title(f"Bias distribution\nstd={bias.std():.2f}", fontsize=11)
    ax6.legend(fontsize=8)

    plt.suptitle(title_main, fontsize=13, fontweight="bold")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  저장: {out_path}")


def main():
    args = parse_args()

    print("=== [V4] 일별 공간 분포 시각화 ===\n")

    print("[1] 데이터 로드...")
    pred, grid = load_data()
    print(f"  predictions: {len(pred):,}행  grid: {len(grid):,}셀")

    # 날짜 선택
    if args.date is None:
        date_str = select_date(pred)
        print(f"  날짜 자동 선택 (고농도 최대): {date_str}")
    else:
        date_str = args.date
        print(f"  날짜 지정: {date_str}")

    # 필터링
    day = filter_day(pred, date_str, args.hour)
    if day.empty:
        print(f"  [오류] {date_str} 에 해당하는 데이터 없음")
        available = sorted(pred["date"].dt.date.unique())
        print(f"  사용 가능한 날짜 (처음 10개): {available[:10]}")
        sys.exit(1)

    hour_str = f"h{args.hour:02d}" if args.hour is not None else "daily"
    print(f"  필터 결과: {len(day):,}행  "
          f"actual_mean={day['road_pm_actual'].mean():.2f}  "
          f"max={day['road_pm_actual'].max():.1f}")

    # 격자별 집계
    print("\n[2] 격자별 집계...")
    stats = compute_cell_stats(day)
    print(f"  집계 셀 수: {len(stats):,}")

    # 출력 경로
    if args.out:
        out_path = args.out
    else:
        fname = f"daily_{date_str.replace('-', '')}_{hour_str}.png"
        out_path = os.path.join(CKPT_DIR, fname)

    # 시각화
    print("\n[3] 시각화 생성...")
    make_daily_figure(stats, grid, date_str, args.hour, out_path)

    print("\n완료.")


if __name__ == "__main__":
    main()
