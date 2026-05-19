import numpy as np
from scipy.spatial.distance import cdist


def compute_route_distance(
    route,
    coords
):

    dist_matrix = cdist(
        coords,
        coords
    )

    total = 0.0

    for i in range(len(route) - 1):

        total += dist_matrix[
            route[i],
            route[i + 1]
        ]

    return total


def compute_total_score(
    route,
    prizes
):

    return prizes[route].sum()


def compute_score_per_distance(
    route,
    coords,
    prizes
):

    dist = compute_route_distance(
        route,
        coords
    )

    score = compute_total_score(
        route,
        prizes
    )

    return score / (dist + 1e-6)