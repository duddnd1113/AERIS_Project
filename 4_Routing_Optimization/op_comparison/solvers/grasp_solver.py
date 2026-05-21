import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


def route_distance(route, dist_matrix):
    if len(route) <= 1:
        return 0.0

    total = 0.0
    for i in range(len(route) - 1):
        total += dist_matrix[route[i], route[i + 1]]

    return total


def greedy_randomized_construction(
    coords,
    prizes,
    budget,
    alpha=0.3,
    start_idx=0
):
    n = len(coords)

    dist_matrix = cdist(coords, coords)

    unvisited = set(range(n))
    unvisited.remove(start_idx)

    route = [start_idx]
    current = start_idx

    while True:

        candidates = []

        for j in unvisited:

            added_cost = dist_matrix[current, j]

            projected = (
                route_distance(route, dist_matrix)
                + added_cost
            )

            if projected <= budget:

                ratio = prizes[j] / (added_cost + 1e-6)

                candidates.append((j, ratio))

        if len(candidates) == 0:
            break

        candidates.sort(key=lambda x: x[1], reverse=True)

        rcl_size = max(1, int(alpha * len(candidates)))

        selected = np.random.choice(
            range(rcl_size)
        )

        next_node = candidates[selected][0]

        route.append(next_node)

        unvisited.remove(next_node)

        current = next_node

    return route


def local_search(route, coords, prizes, budget):

    dist_matrix = cdist(coords, coords)

    improved = True

    while improved:

        improved = False

        for i in range(1, len(route) - 1):

            for j in range(i + 1, len(route)):

                new_route = route.copy()

                new_route[i:j] = reversed(
                    new_route[i:j]
                )

                old_dist = route_distance(
                    route,
                    dist_matrix
                )

                new_dist = route_distance(
                    new_route,
                    dist_matrix
                )

                if (
                    new_dist < old_dist
                    and new_dist <= budget
                ):
                    route = new_route
                    improved = True

    return route


def solve_grasp(
    df,
    budget=5.0,
    iterations=50,
    alpha=0.3
):

    coords = df[["x", "y"]].values
    prizes = df["score"].values

    best_route = None
    best_score = -1

    for _ in range(iterations):

        route = greedy_randomized_construction(
            coords,
            prizes,
            budget,
            alpha
        )

        route = local_search(
            route,
            coords,
            prizes,
            budget
        )

        total_score = prizes[route].sum()

        if total_score > best_score:
            best_score = total_score
            best_route = route

    return best_route