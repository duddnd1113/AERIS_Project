import pandas as pd

from evaluation.kpi import (
    compute_route_distance,
    compute_total_score,
    compute_score_per_distance
)


def compare_routes(
    grasp_route,
    rl_route,
    df
):

    coords = df[["x", "y"]].values
    prizes = df["score"].values

    results = []

    for name, route in [
        ("GRASP", grasp_route),
        ("ActiveSearch", rl_route)
    ]:

        results.append({

            "method": name,

            "distance":
                compute_route_distance(
                    route,
                    coords
                ),

            "score":
                compute_total_score(
                    route,
                    prizes
                ),

            "score_per_distance":
                compute_score_per_distance(
                    route,
                    coords,
                    prizes
                ),

            "visited_nodes":
                len(route)
        })

    return pd.DataFrame(results)