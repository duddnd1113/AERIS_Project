"""
[V4] 시각화
  1. 공간 분포 히트맵: 격자별 실제 / 예측 road_pm 평균 비교
  2. Scatter plot: 예측 vs 실제 (고농도 구간 강조)
  3. 고농도 이벤트 분포 히트맵 (road_pm > 50)

실행:
    python3 step_visualize.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

# Korean font — use if available, otherwise fall back to English labels
import matplotlib.font_manager as fm
_korean_fonts = [f.name for f in fm.fontManager.ttflist
                 if any(k in f.name for k in ("Gothic", "Nanum", "Malgun", "AppleGothic"))]
if _korean_fonts:
    plt.rcParams["font.family"] = _korean_fonts[0]
else:
    plt.rcParams["font.family"] = "DejaVu Sans"

from config import PRED_CSV, GRID_CSV, HEATMAP_PNG, SCATTER_PNG, CKPT_DIR, RESULTS_JSON


# ── 헬퍼 ─────────────────────────────────────────────────────────────────
def load_data():
    pred = pd.read_csv(PRED_CSV, parse_dates=["date"])
    grid = pd.read_csv(GRID_CSV)[["CELL_ID", "lat", "lon"]]
    return pred, grid


def cell_stats(pred: pd.DataFrame) -> pd.DataFrame:
    """격자별 실제/예측 road_pm 통계 계산."""
    grp = pred.groupby("CELL_ID").agg(
        actual_mean   =("road_pm_actual", "mean"),
        pred_mean     =("road_pm_pred",   "mean"),
        actual_p95    =("road_pm_actual", lambda x: np.percentile(x, 95)),
        pred_p95      =("road_pm_pred",   lambda x: np.percentile(x, 95)),
        n             =("road_pm_actual", "count"),
    ).reset_index()
    return grp


def scatter_plot(y_true, y_pred, mae, r2, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ── All samples ───────────────────────────────────────────────────────
    ax = axes[0]
    c  = np.clip(y_true, 0, 80)
    sc = ax.scatter(y_true, y_pred, c=c, cmap="YlOrRd",
                    alpha=0.25, s=3, rasterized=True)
    max_val = max(float(y_true.max()), float(y_pred.max()), 10)
    ax.plot([0, max_val], [0, max_val], "k--", lw=1.2, label="y=x")
    ax.set_xlim(0, max_val); ax.set_ylim(0, max_val)
    ax.set_xlabel("Actual road_pm (ug/m3)")
    ax.set_ylabel("Predicted road_pm (ug/m3)")
    ax.set_title(f"All  MAE={mae:.3f}  R2={r2:.4f}")
    ax.legend(fontsize=9)
    plt.colorbar(sc, ax=ax, label="Actual road_pm")

    # ── High-PM scatter (> 30) ────────────────────────────────────────────
    ax2 = axes[1]
    mask = y_true > 30
    if mask.sum() > 10:
        hi_mae = float(np.mean(np.abs(y_true[mask] - y_pred[mask])))
        hi_r2  = float(1 - np.sum((y_true[mask]-y_pred[mask])**2)
                           / np.sum((y_true[mask]-y_true[mask].mean())**2))
        c2 = np.clip(y_true[mask], 30, 150)
        sc2 = ax2.scatter(y_true[mask], y_pred[mask], c=c2,
                          cmap="hot_r", alpha=0.4, s=6, rasterized=True)
        hi_max = max(float(y_true[mask].max()), float(y_pred[mask].max()), 50)
        ax2.plot([0, hi_max], [0, hi_max], "k--", lw=1.2)
        ax2.set_xlim(0, hi_max); ax2.set_ylim(0, hi_max)
        ax2.set_xlabel("Actual road_pm (ug/m3)")
        ax2.set_ylabel("Predicted road_pm (ug/m3)")
        ax2.set_title(f"High-PM > 30  n={mask.sum():,}  MAE={hi_mae:.3f}  R2={hi_r2:.4f}")
        plt.colorbar(sc2, ax=ax2, label="Actual road_pm")
    else:
        ax2.set_visible(False)

    plt.suptitle("V4 Two-Stage (log1p + construction feat)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved: {out_path}")


def heatmap_plot(stats: pd.DataFrame, grid: pd.DataFrame, out_path: str):
    merged = stats.merge(grid, on="CELL_ID", how="inner")
    merged = merged.dropna(subset=["lat", "lon", "actual_mean", "pred_mean"])
    if merged.empty:
        print("  [경고] 격자 좌표 병합 결과 없음 — 히트맵 생성 생략")
        return

    # 공통 컬러 범위 (log scale)
    vmin = max(merged[["actual_mean", "pred_mean"]].min().min(), 0.5)
    vmax = merged[["actual_mean", "pred_mean"]].max().max()
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)

    fig = plt.figure(figsize=(18, 14))
    gs  = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    seoul_lon = (126.75, 127.18)
    seoul_lat = (37.43,  37.70)

    def draw_map(ax, col, title, use_norm=True):
        kw = dict(norm=norm) if use_norm else dict(vmin=0, vmax=vmax * 0.5)
        sc = ax.scatter(
            merged["lon"], merged["lat"],
            c=merged[col], cmap="YlOrRd", s=4, alpha=0.85,
            rasterized=True, **kw,
        )
        ax.set_xlim(*seoul_lon)
        ax.set_ylim(*seoul_lat)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Longitude", fontsize=9)
        ax.set_ylabel("Latitude", fontsize=9)
        ax.tick_params(labelsize=7)
        cb = plt.colorbar(sc, ax=ax, shrink=0.85)
        cb.set_label("road_pm (ug/m3)", fontsize=8)
        return sc

    # Mean actual / predicted
    ax1 = fig.add_subplot(gs[0, 0])
    draw_map(ax1, "actual_mean", "Actual road_pm (mean)")

    ax2 = fig.add_subplot(gs[0, 1])
    draw_map(ax2, "pred_mean", "Predicted road_pm (mean)")

    # Bias map
    ax3 = fig.add_subplot(gs[0, 2])
    err = merged["pred_mean"] - merged["actual_mean"]
    vlim = float(np.percentile(np.abs(err), 95))
    sc3  = ax3.scatter(
        merged["lon"], merged["lat"],
        c=err, cmap="RdBu_r", s=4, alpha=0.85,
        vmin=-vlim, vmax=vlim, rasterized=True,
    )
    ax3.set_xlim(*seoul_lon); ax3.set_ylim(*seoul_lat)
    ax3.set_title("Bias (pred - actual)", fontsize=11)
    ax3.set_xlabel("Longitude", fontsize=9); ax3.set_ylabel("Latitude", fontsize=9)
    ax3.tick_params(labelsize=7)
    cb3 = plt.colorbar(sc3, ax=ax3, shrink=0.85)
    cb3.set_label("delta road_pm (ug/m3)", fontsize=8)

    # p95 actual / predicted
    ax4 = fig.add_subplot(gs[1, 0])
    draw_map(ax4, "actual_p95", "Actual road_pm p95 (high-PM distribution)", use_norm=False)

    ax5 = fig.add_subplot(gs[1, 1])
    draw_map(ax5, "pred_p95", "Predicted road_pm p95 (high-PM distribution)", use_norm=False)

    # Bias histogram
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.hist(err, bins=60, color="steelblue", edgecolor="white", linewidth=0.3)
    ax6.axvline(0, color="red", linestyle="--", linewidth=1.2)
    ax6.set_xlabel("Pred - Actual road_pm (ug/m3)", fontsize=9)
    ax6.set_ylabel("Grid count", fontsize=9)
    ax6.set_title(
        f"Per-grid bias distribution\nbias={err.mean():.2f}  std={err.std():.2f}", fontsize=10
    )

    plt.suptitle(
        "V4 Two-Stage (log1p + construction feat) — Spatial road_pm distribution",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved: {out_path}")


def main():
    print("=== [V4] 시각화 ===\n")

    print("[1] 예측 데이터 로드...")
    pred, grid = load_data()
    print(f"  predictions: {len(pred):,} rows  grid: {len(grid):,} cells")

    y_true = pred["road_pm_actual"].values
    y_pred = pred["road_pm_pred"].values
    mae    = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2     = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    print(f"  MAE={mae:.4f}  R2={r2:.4f}")

    print("\n[2] Scatter plot...")
    scatter_plot(y_true, y_pred, mae, r2, SCATTER_PNG)

    print("\n[3] Per-cell statistics...")
    stats = cell_stats(pred)
    print(f"  cells: {len(stats):,}")
    print(f"  actual_mean range: {stats['actual_mean'].min():.2f} ~ "
          f"{stats['actual_mean'].max():.2f} ug/m3")
    print(f"  pred_mean   range: {stats['pred_mean'].min():.2f} ~ "
          f"{stats['pred_mean'].max():.2f} ug/m3")

    print("\n[4] Spatial heatmap...")
    heatmap_plot(stats, grid, HEATMAP_PNG)

    print("\n완료.")


if __name__ == "__main__":
    main()
