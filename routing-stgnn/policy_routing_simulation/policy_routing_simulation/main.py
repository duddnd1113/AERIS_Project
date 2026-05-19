import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class Config:
    # Grid / city simulation
    n_tiles: int = 18_000
    city_width_km: float = 45.0
    city_height_km: float = 40.0
    seed: int = 42

    # Policy settings
    n_observed_current: int = 250       # current policy can watch only a few tiles
    threshold_pm: float = 150.0         # dispatch threshold for current policy
    top_k_predictive: int = 120         # ours selects top-k predicted priority tiles

    # Routing / OP-style budget
    max_route_km: float = 80.0
    depot_x: float = 22.5
    depot_y: float = 20.0

    # Benefit definition for the route solver
    # "pm" = maximize PM removed
    # "exposure" = maximize PM * population
    route_benefit: str = "exposure"

    # Prediction noise
    pred_noise_std: float = 12.0


def make_synthetic_grid(cfg: Config) -> pd.DataFrame:
    """
    Creates synthetic citywide grid data:
    - x, y coordinates
    - true PM score
    - predicted PM score
    - population
    - exposure score = PM * population

    The PM surface has broad spatial trends + hotspots + random noise.
    """
    rng = np.random.default_rng(cfg.seed)

    x = rng.uniform(0, cfg.city_width_km, cfg.n_tiles)
    y = rng.uniform(0, cfg.city_height_km, cfg.n_tiles)

    # Broad spatial pollution surface
    broad_trend = (
        45
        + 25 * np.sin(x / cfg.city_width_km * np.pi)
        + 18 * np.cos(y / cfg.city_height_km * 2 * np.pi)
        + 12 * (x / cfg.city_width_km)
    )

    # Hotspots: (x, y, intensity, radius)
    hotspots = [
        (10, 12, 110, 4.0),
        (30, 28, 95, 5.5),
        (37, 10, 80, 4.5),
        (18, 32, 70, 6.0),
    ]

    pm = broad_trend.copy()
    for hx, hy, amp, sigma in hotspots:
        d2 = (x - hx) ** 2 + (y - hy) ** 2
        pm += amp * np.exp(-d2 / (2 * sigma ** 2))

    pm += rng.normal(0, 8, cfg.n_tiles)
    pm = np.clip(pm, 5, None)

    # Population: concentrated in a few urban centers
    pop = (
        150
        + 1400 * np.exp(-((x - 22) ** 2 + (y - 20) ** 2) / (2 * 8 ** 2))
        + 800 * np.exp(-((x - 34) ** 2 + (y - 30) ** 2) / (2 * 6 ** 2))
        + 500 * rng.random(cfg.n_tiles)
    )
    pop = np.clip(pop, 1, None)

    # Predicted PM = true PM + model noise
    pred_pm = pm + rng.normal(0, cfg.pred_noise_std, cfg.n_tiles)
    pred_pm = np.clip(pred_pm, 0, None)

    df = pd.DataFrame({
        "tile_id": np.arange(cfg.n_tiles),
        "x": x,
        "y": y,
        "true_pm": pm,
        "pred_pm": pred_pm,
        "population": pop,
    })
    df["true_exposure"] = df["true_pm"] * df["population"]
    df["pred_exposure"] = df["pred_pm"] * df["population"]
    return df


def euclidean_km(a_x, a_y, b_x, b_y) -> float:
    return float(np.sqrt((a_x - b_x) ** 2 + (a_y - b_y) ** 2))


def greedy_orienteering_solver(
    candidates: pd.DataFrame,
    cfg: Config,
    benefit_col: str,
) -> tuple[list[int], float]:
    """
    Simple OP-style greedy solver.

    At each step, select the candidate with the best marginal benefit per added km,
    while ensuring the route can still return to the depot within max_route_km.

    This is not an exact OP solver, but it is useful for fair simulation because
    both policies use the exact same routing logic.
    """
    remaining = candidates.copy()
    visited = []

    cur_x, cur_y = cfg.depot_x, cfg.depot_y
    total_km = 0.0

    while len(remaining) > 0:
        best_idx = None
        best_score = -np.inf
        best_added_km = None

        for idx, row in remaining.iterrows():
            go_km = euclidean_km(cur_x, cur_y, row["x"], row["y"])
            return_km = euclidean_km(row["x"], row["y"], cfg.depot_x, cfg.depot_y)

            # Must be able to visit this grid and return to depot
            if total_km + go_km + return_km > cfg.max_route_km:
                continue

            score = row[benefit_col] / max(go_km, 1e-6)

            if score > best_score:
                best_score = score
                best_idx = idx
                best_added_km = go_km

        if best_idx is None:
            break

        selected = remaining.loc[best_idx]
        visited.append(int(selected["tile_id"]))
        total_km += best_added_km
        cur_x, cur_y = float(selected["x"]), float(selected["y"])
        remaining = remaining.drop(index=best_idx)

    # Return to depot if at least one tile was visited
    if visited:
        total_km += euclidean_km(cur_x, cur_y, cfg.depot_x, cfg.depot_y)

    return visited, total_km


def current_reactive_policy(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Current-Reactive:
    - only a limited number of tiles can be observed
    - dispatch candidates are observed tiles whose true PM exceeds threshold
    """
    rng = np.random.default_rng(cfg.seed + 100)
    observed_ids = rng.choice(df["tile_id"].values, size=cfg.n_observed_current, replace=False)
    observed = df[df["tile_id"].isin(observed_ids)].copy()

    candidates = observed[observed["true_pm"] >= cfg.threshold_pm].copy()

    # For current policy, it only knows observed PM.
    # But actual KPI evaluation still uses true PM and true exposure.
    return candidates


def ours_predictive_policy(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Ours-Predictive:
    - can score the whole city grid
    - selects top-k tiles based on predicted exposure or predicted PM
    """
    if cfg.route_benefit == "exposure":
        rank_col = "pred_exposure"
    else:
        rank_col = "pred_pm"

    candidates = df.nlargest(cfg.top_k_predictive, rank_col).copy()
    return candidates


def compute_kpis(
    name: str,
    df: pd.DataFrame,
    visited_ids: list[int],
    route_km: float,
    high_risk_ids: set[int],
) -> dict:
    visited_df = df[df["tile_id"].isin(visited_ids)].copy()
    cleaned_grids = len(visited_df)

    pollution_reduction = visited_df["true_pm"].sum()
    exposure_reduction = visited_df["true_exposure"].sum()

    cleaned_high_risk = visited_df["tile_id"].isin(high_risk_ids).sum()
    high_risk_hit_rate = cleaned_high_risk / cleaned_grids if cleaned_grids > 0 else 0.0

    route_efficiency = exposure_reduction / route_km if route_km > 0 else 0.0

    return {
        "policy": name,
        "cleaned_grids": cleaned_grids,
        "route_km": route_km,
        "pollution_reduction_sum_PM": pollution_reduction,
        "exposure_reduction_sum_PMxPop": exposure_reduction,
        "high_risk_hit_rate": high_risk_hit_rate,
        "cleaned_high_risk_grids": int(cleaned_high_risk),
        "route_efficiency_exposure_per_km": route_efficiency,
    }


def plot_routes(df, routes, cfg, out_dir):
    plt.figure(figsize=(10, 8))
    sc = plt.scatter(
        df["x"],
        df["y"],
        c=df["true_pm"],
        s=4,
        alpha=0.35,
    )
    plt.colorbar(sc, label="True PM")

    plt.scatter([cfg.depot_x], [cfg.depot_y], marker="s", s=140, label="Depot")

    for policy_name, visited_ids in routes.items():
        route_df = df[df["tile_id"].isin(visited_ids)].copy()
        if len(route_df) == 0:
            continue

        # Preserve visit order
        order_map = {tile_id: i for i, tile_id in enumerate(visited_ids)}
        route_df["visit_order"] = route_df["tile_id"].map(order_map)
        route_df = route_df.sort_values("visit_order")

        xs = [cfg.depot_x] + route_df["x"].tolist() + [cfg.depot_x]
        ys = [cfg.depot_y] + route_df["y"].tolist() + [cfg.depot_y]

        plt.plot(xs, ys, linewidth=2, label=policy_name)
        plt.scatter(route_df["x"], route_df["y"], s=35)

    plt.title("Route Comparison")
    plt.xlabel("x coordinate (km)")
    plt.ylabel("y coordinate (km)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "route_comparison.png"), dpi=200)
    plt.close()


def plot_kpis(kpi_df, out_dir):
    metrics = [
        "pollution_reduction_sum_PM",
        "exposure_reduction_sum_PMxPop",
        "high_risk_hit_rate",
        "route_efficiency_exposure_per_km",
    ]

    for metric in metrics:
        plt.figure(figsize=(7, 5))
        plt.bar(kpi_df["policy"], kpi_df[metric])
        plt.title(metric)
        plt.ylabel(metric)
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"kpi_{metric}.png"), dpi=200)
        plt.close()


def run_experiment():
    cfg = Config()
    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)

    df = make_synthetic_grid(cfg)

    # Define true high-risk grids for evaluation.
    # Here: top 5% by true exposure.
    high_risk_cutoff = df["true_exposure"].quantile(0.95)
    high_risk_ids = set(df.loc[df["true_exposure"] >= high_risk_cutoff, "tile_id"].astype(int))

    if cfg.route_benefit == "exposure":
        current_benefit_col = "true_exposure"
        ours_benefit_col = "pred_exposure"
    else:
        current_benefit_col = "true_pm"
        ours_benefit_col = "pred_pm"

    current_candidates = current_reactive_policy(df, cfg)
    ours_candidates = ours_predictive_policy(df, cfg)

    current_visited, current_km = greedy_orienteering_solver(
        current_candidates,
        cfg,
        benefit_col=current_benefit_col,
    )
    ours_visited, ours_km = greedy_orienteering_solver(
        ours_candidates,
        cfg,
        benefit_col=ours_benefit_col,
    )

    kpis = [
        compute_kpis("Current-Reactive", df, current_visited, current_km, high_risk_ids),
        compute_kpis("Ours-Predictive", df, ours_visited, ours_km, high_risk_ids),
    ]

    kpi_df = pd.DataFrame(kpis)

    df.to_csv(os.path.join(out_dir, "synthetic_grid.csv"), index=False)
    current_candidates.to_csv(os.path.join(out_dir, "current_candidates.csv"), index=False)
    ours_candidates.to_csv(os.path.join(out_dir, "ours_candidates.csv"), index=False)
    kpi_df.to_csv(os.path.join(out_dir, "kpi_comparison.csv"), index=False)

    plot_routes(
        df,
        {
            "Current-Reactive": current_visited,
            "Ours-Predictive": ours_visited,
        },
        cfg,
        out_dir,
    )

    plot_kpis(kpi_df, out_dir)

    print("\n=== KPI Comparison ===")
    print(kpi_df.to_string(index=False))

    print("\nSaved outputs to:", out_dir)


if __name__ == "__main__":
    run_experiment()
