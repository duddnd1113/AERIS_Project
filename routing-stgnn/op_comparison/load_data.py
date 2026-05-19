import os
import numpy as np
import pandas as pd


PM_FILE = "pm_filtered.csv"   # your Excel file
PM_COL = "재비산먼지 평균농도(㎍/㎥)"


def load_pm_values(pm_file=PM_FILE, pm_col=PM_COL):
    pm_df = pd.read_csv(pm_file)

    pm_values = pm_df[pm_col].dropna().values
    return pm_values


def generate_synthetic_snapshot(
    pm_values,
    n_nodes=100,
    city_width=1.0,
    city_height=1.0,
    seed=None,
    use_hotspots=True
):
    rng = np.random.default_rng(seed)

    x_coords = rng.uniform(0, city_width, n_nodes)
    y_coords = rng.uniform(0, city_height, n_nodes)

    sampled_pm = rng.choice(
        pm_values,
        size=n_nodes,
        replace=True
    )

    snapshot = pd.DataFrame({
        "x": x_coords,
        "y": y_coords,
        "score": sampled_pm
    })

    if use_hotspots:
        hotspot_centers = [
            (0.25, 0.30),
            (0.70, 0.60),
            (0.50, 0.85)
        ]

        for hx, hy in hotspot_centers:
            dist = np.sqrt(
                (snapshot["x"] - hx) ** 2
                + (snapshot["y"] - hy) ** 2
            )

            hotspot_effect = 10 * np.exp(
                -(dist ** 2) / (2 * 0.12 ** 2)
            )

            snapshot["score"] += hotspot_effect

    snapshot["score"] = np.clip(
        snapshot["score"],
        0,
        None
    )

    return snapshot


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    pm_values = load_pm_values()

    snapshot = generate_synthetic_snapshot(
        pm_values,
        n_nodes=100,
        seed=42
    )

    snapshot.to_csv(
        "data/snapshot.csv",
        index=False
    )

    print(snapshot.head())
    print(snapshot.describe())