"""
OPTopKProblemDef.py
====================
Problem definition utilities for the Orienteering Problem (OP)
on a scored grid snapshot.

Key difference from CVRP:
    - Scores become PRIZES (normalized to [0,1]), not demands.
    - No depot or vehicle capacity — only a travel budget T.
    - Goal: maximize total prize collected within budget T.
"""

import torch


def select_topk_cells(coords: torch.Tensor, scores: torch.Tensor, top_k: int):
    """Select top-k cells by score from the full grid.

    Args:
        coords: (batch, total_grid_size, 2)
        scores: (batch, total_grid_size)
        top_k:  int

    Returns:
        selected_coords:  (batch, top_k, 2)
        selected_prizes:  (batch, top_k)  -- normalized to [0, 1]
        selected_indices: (batch, top_k)  -- original grid indices
    """
    _, selected_indices = torch.topk(scores, k=top_k, dim=1, largest=True, sorted=True)

    gather_idx = selected_indices[:, :, None].expand(-1, -1, 2)
    selected_coords = coords.gather(dim=1, index=gather_idx)

    # Gather raw scores, normalize per-batch to [0, 1] → prizes
    raw = scores.gather(dim=1, index=selected_indices)
    s_min = raw.min(dim=1, keepdim=True).values
    s_max = raw.max(dim=1, keepdim=True).values
    selected_prizes = (raw - s_min) / (s_max - s_min + 1e-8)

    return selected_coords, selected_prizes, selected_indices


def get_random_scored_grids(batch_size: int, total_grid_size: int, device=None):
    """Generate a synthetic scored grid for training / debugging.

    Returns:
        coords: (batch, total_grid_size, 2)  -- normalized [0,1]
        scores: (batch, total_grid_size)     -- positive, unnormalized
    """
    grid_size = int(total_grid_size ** 0.5)
    if grid_size * grid_size != total_grid_size:
        raise ValueError(
            f"total_grid_size must be a perfect square. "
            f"Got {total_grid_size}. Try e.g. 17956 (134²) or 18225 (135²)."
        )

    x = torch.linspace(0, 1, grid_size, device=device)
    y = torch.linspace(0, 1, grid_size, device=device)
    try:
        xv, yv = torch.meshgrid(x, y, indexing='ij')
    except TypeError:
        xv, yv = torch.meshgrid(x, y)

    base_coords = torch.stack([xv.reshape(-1), yv.reshape(-1)], dim=1)
    coords = base_coords[None].expand(batch_size, total_grid_size, 2).clone()

    # Hotspot + noise — replace with real scores at inference
    center = torch.tensor([0.55, 0.45], device=device)
    dist = ((coords - center) ** 2).sum(dim=2).sqrt()
    hotspot = torch.exp(-8.0 * dist)
    noise = 0.25 * torch.rand(batch_size, total_grid_size, device=device)
    scores = (hotspot + noise).clamp(min=0.0)

    return coords, scores


def augment_xy_data_by_8_fold(xy_data: torch.Tensor):
    """8-fold coordinate augmentation (POMO paper, Table 1).

    Input:  (batch, N, 2)
    Output: (8*batch, N, 2)
    """
    x = xy_data[:, :, 0:1]
    y = xy_data[:, :, 1:2]
    return torch.cat([
        torch.cat([x,     y    ], dim=2),
        torch.cat([1 - x, y    ], dim=2),
        torch.cat([x,     1 - y], dim=2),
        torch.cat([1 - x, 1 - y], dim=2),
        torch.cat([y,     x    ], dim=2),
        torch.cat([1 - y, x    ], dim=2),
        torch.cat([y,     1 - x], dim=2),
        torch.cat([1 - y, 1 - x], dim=2),
    ], dim=0)
