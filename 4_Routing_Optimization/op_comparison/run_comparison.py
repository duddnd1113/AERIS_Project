import os
import pandas as pd

from load_data import (
    load_pm_values,
    generate_synthetic_snapshot
)

from solvers.grasp_solver import solve_grasp
from solvers.active_search_solver import solve_active_search
from evaluation.compare import compare_routes


N_SIM = 20
N_NODES = 100
BUDGET = 3.0
EPOCHS = 300

OUTPUT_DIR = "outputs"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/snapshots", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/routes", exist_ok=True)

    pm_values = load_pm_values()

    all_results = []

    for sim in range(N_SIM):
        print(f"\n==============================")
        print(f"Simulation {sim + 1}/{N_SIM}")
        print(f"==============================")

        df = generate_synthetic_snapshot(
            pm_values=pm_values,
            n_nodes=N_NODES,
            city_width=1.0,
            city_height=1.0,
            seed=sim,
            use_hotspots=True
        )

        snapshot_path = f"{OUTPUT_DIR}/snapshots/snapshot_sim_{sim}.csv"
        df.to_csv(snapshot_path, index=False)

        grasp_route = solve_grasp(
            df,
            budget=BUDGET,
            iterations=50,
            alpha=0.3
        )

    
        rl_route = solve_active_search(
    df,
    budget=BUDGET,
    epochs=EPOCHS,
    lr=1e-4,
    device="cpu",
    top_k=N_NODES,
    sim_id=sim,
    log_interval=10
)

        result = compare_routes(
            grasp_route,
            rl_route,
            df
        )

        result.insert(0, "sim", sim)

        all_results.append(result)

        pd.DataFrame({
            "node": grasp_route
        }).to_csv(
            f"{OUTPUT_DIR}/routes/grasp_route_sim_{sim}.csv",
            index=False
        )

        pd.DataFrame({
            "node": rl_route
        }).to_csv(
            f"{OUTPUT_DIR}/routes/active_search_route_sim_{sim}.csv",
            index=False
        )

    all_results_df = pd.concat(
        all_results,
        ignore_index=True
    )

    all_results_df.to_csv(
        f"{OUTPUT_DIR}/kpi_results_all.csv",
        index=False
    )

    summary = (
        all_results_df
        .groupby("method")
        .agg({
            "distance": ["mean", "std"],
            "score": ["mean", "std"],
            "score_per_distance": ["mean", "std"],
            "visited_nodes": ["mean", "std"]
        })
    )

    summary.to_csv(
        f"{OUTPUT_DIR}/kpi_summary.csv"
    )

    print("\n=== ALL RESULTS ===")
    print(all_results_df)

    print("\n=== SUMMARY ===")
    print(summary)


if __name__ == "__main__":
    main()