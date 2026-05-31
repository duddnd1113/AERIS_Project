#!/usr/bin/env python3
"""Create Top-10 configuration ranking plots for Appendix C4 results."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "all_results.csv"

GRAPH_LABELS = {
    "climatological": "Climatological",
    "static": "Static",
    "soft_dynamic": "Soft dynamic",
}
GRAPH_COLORS = {
    "climatological": "#6A7FDB",
    "static": "#2C8C99",
    "soft_dynamic": "#D66B43",
}
METRIC_LABELS = {
    "mae": "MAE",
    "rmse": "RMSE",
}


def scenario_key(value: str) -> int:
    return int(str(value).replace("S", ""))


def load_results() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    df["scenario_order"] = df["scenario_short"].map(scenario_key)
    df["graph_label"] = df["graph_mode"].map(GRAPH_LABELS).fillna(df["graph_mode"])
    df["configuration"] = (
        "W" + df["window"].astype(str)
        + " | " + df["scenario_short"]
        + " | " + df["graph_label"]
    )
    return df.sort_values(["window", "scenario_order", "graph_mode"]).reset_index(drop=True)


def top10(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    ranked = df.sort_values([metric, "window", "scenario_order", "graph_mode"], ascending=True).head(10).copy()
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    return ranked


def plot_top10(ranked: pd.DataFrame, metric: str) -> Path:
    metric_label = METRIC_LABELS[metric]
    other_metric = "rmse" if metric == "mae" else "mae"
    other_label = METRIC_LABELS[other_metric]
    out_path = HERE / f"top10_configurations_by_{metric}.png"

    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.labelsize": 15,
            "axes.titlesize": 20,
            "figure.titlesize": 22,
        }
    )

    plot_df = ranked.sort_values("rank", ascending=False).copy()
    labels = [f"#{row.rank:02d}  {row.configuration}" for row in plot_df.itertuples()]
    colors = [GRAPH_COLORS.get(mode, "#777777") for mode in plot_df["graph_mode"]]

    fig_height = max(8.0, 0.42 * len(plot_df) + 2.2)
    fig, ax = plt.subplots(figsize=(16, fig_height + 1.0), constrained_layout=True)
    bars = ax.barh(labels, plot_df[metric], color=colors, edgecolor="white", linewidth=0.8)

    min_value = plot_df[metric].min()
    max_value = plot_df[metric].max()
    span = max_value - min_value
    pad = max(span * 0.18, max_value * 0.015)
    ax.set_xlim(max(0, min_value - pad), max_value + pad * 1.75)

    for bar, row in zip(bars, plot_df.itertuples()):
        x = bar.get_width()
        y = bar.get_y() + bar.get_height() / 2
        ax.text(
            x + pad * 0.12,
            y,
            f"{metric_label} {getattr(row, metric):.3f} | {other_label} {getattr(row, other_metric):.3f}",
            va="center",
            ha="left",
            fontsize=13,
            color="#303236",
        )

    ax.set_title(f"Top 10 Configurations Ranked by {metric_label}")
    ax.set_xlabel(f"{metric_label} (lower is better)")
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=13)
    ax.tick_params(axis="x", labelsize=13)
    ax.grid(axis="x", color="#d8dce2", linewidth=0.8)
    ax.grid(axis="y", visible=False)

    handles = [
        plt.Line2D([0], [0], marker="s", linestyle="", markersize=10, color=color, label=label)
        for key, label in GRAPH_LABELS.items()
        for color in [GRAPH_COLORS[key]]
    ]
    ax.legend(handles=handles, loc="lower right", frameon=True, title="Graph mode", fontsize=13, title_fontsize=14)

    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_rank_table(ranked: pd.DataFrame, metric: str) -> Path:
    out_path = HERE / f"top10_configurations_by_{metric}.csv"
    columns = [
        "rank",
        "window",
        "scenario_short",
        "graph_mode",
        "mae",
        "rmse",
        "best_val_loss",
        "elapsed_min",
        "n_features",
        "n_nodes",
        "n_edges",
        "scenario",
    ]
    ranked[columns].to_csv(out_path, index=False)
    return out_path


def main() -> None:
    df = load_results()
    for metric in ["mae", "rmse"]:
        ranked = top10(df, metric)
        csv_path = save_rank_table(ranked, metric)
        png_path = plot_top10(ranked, metric)
        best = ranked.iloc[0]
        print(f"Saved: {png_path}")
        print(f"Saved: {csv_path}")
        print(
            f"Best by {METRIC_LABELS[metric]}: "
            f"#{int(best['rank'])} W{int(best['window'])} {best['scenario_short']} "
            f"{best['graph_mode']} | MAE={best['mae']:.4f}, RMSE={best['rmse']:.4f}"
        )


if __name__ == "__main__":
    main()
