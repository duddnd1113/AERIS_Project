"""
OPTopKEnvironment.py
=====================
POMO-style Orienteering Problem (OP) environment for a top-k scored grid.

Key differences from CVRP environment:
    - No depot, no vehicle capacity.
    - Budget T = max allowed travel distance.
    - At each step, nodes infeasible if:
          dist(current → node) + dist(node → start) > remaining_budget
      (must be able to return to start after visiting node)
    - Episode ends when no feasible unvisited nodes remain.
    - Reward = sum of prizes collected (positive), NOT negative distance.
    - Node input features are (x, y, prize) — 3-dimensional.

POMO multiple starts:
    - Trajectory i starts at node i (0-indexed among top_k nodes).
    - All top_k starts run in parallel.
"""

from dataclasses import dataclass
import torch

from rl.OPTopKProblemDef import (
    select_topk_cells,
    get_random_scored_grids,
    augment_xy_data_by_8_fold,
)


@dataclass
class Reset_State:
    node_xy:          torch.Tensor = None   # (batch, top_k, 2)
    node_prize:       torch.Tensor = None   # (batch, top_k)      normalized [0,1]
    node_xy_prize:    torch.Tensor = None   # (batch, top_k, 3)   input to encoder
    selected_scores:  torch.Tensor = None   # (batch, top_k)      same as node_prize
    selected_indices: torch.Tensor = None   # (batch, top_k)      original grid idx


@dataclass
class Step_State:
    BATCH_IDX:      torch.Tensor = None
    POMO_IDX:       torch.Tensor = None
    current_node:   torch.Tensor = None     # (batch, pomo)
    ninf_mask:      torch.Tensor = None     # (batch, pomo, top_k)
    remaining_budget: torch.Tensor = None   # (batch, pomo)
    finished:       torch.Tensor = None     # (batch, pomo)  bool


class OPTopKEnv:
    """
    Orienteering Problem environment for POMO active search on a scored grid.

    Pipeline:
        total_grid_size scored cells
            → select top_k by score
            → normalize scores to prizes in [0,1]
            → OP: maximize prize subject to travel budget T
    """

    def __init__(self, **env_params):
        self.env_params    = env_params
        self.total_grid_size = env_params.get('total_grid_size', 18225)
        self.top_k         = env_params['top_k']           # 300
        self.problem_size  = self.top_k
        self.pomo_size     = env_params.get('pomo_size', self.top_k)
        self.budget        = env_params['budget']           # e.g. 3.0

        if self.pomo_size != self.problem_size:
            raise ValueError('For OP POMO, set pomo_size == top_k.')

        # populated by load_problems
        self.batch_size       = None
        self.BATCH_IDX        = None
        self.POMO_IDX         = None

        self.node_xy          = None   # (batch, top_k, 2)
        self.node_prize       = None   # (batch, top_k)
        self.node_xy_prize    = None   # (batch, top_k, 3)
        self.selected_indices = None   # (batch, top_k)

        # episode state
        self.selected_count   = None
        self.current_node     = None
        self.selected_node_list = None
        self.collected_prize  = None   # (batch, pomo)
        self.remaining_budget = None   # (batch, pomo)
        self.visited_mask     = None   # (batch, pomo, top_k)  bool
        self.ninf_mask        = None   # (batch, pomo, top_k)
        self.finished         = None   # (batch, pomo)  bool

        # start coords per pomo trajectory  (batch, pomo, 2)
        self.start_xy         = None

        self.reset_state = Reset_State()
        self.step_state  = Step_State()

    # ------------------------------------------------------------------
    # load_problems
    # ------------------------------------------------------------------
    def load_problems(self, batch_size, aug_factor=1, coords=None, scores=None):
        """Prepare one batch of OP instances.

        Args:
            batch_size: int
            aug_factor: 1 or 8
            coords:  optional (batch, total_grid_size, 2). None → synthetic.
            scores:  optional (batch, total_grid_size).    None → synthetic.
        """
        self.batch_size = batch_size

        if coords is None or scores is None:
            coords, scores = get_random_scored_grids(
                batch_size, self.total_grid_size,
                device=torch.empty(0).device
            )
        else:
            if coords.dim() != 3 or scores.dim() != 2:
                raise ValueError('coords must be (batch, N, 2); scores (batch, N).')
            if coords.size(0) != batch_size or scores.size(0) != batch_size:
                raise ValueError('Batch dimension mismatch.')
            if coords.size(1) < self.top_k:
                raise ValueError('Need at least top_k candidate cells.')

        device = coords.device

        # ── Select top-k ──────────────────────────────────────────────
        node_xy, node_prize, selected_indices = select_topk_cells(
            coords, scores, self.top_k
        )
        # node_xy:    (batch, top_k, 2)
        # node_prize: (batch, top_k)  in [0,1]

        # ── 8-fold augmentation ───────────────────────────────────────
        if aug_factor > 1:
            if aug_factor != 8:
                raise NotImplementedError('Only aug_factor=8 is supported.')
            self.batch_size *= 8
            node_xy         = augment_xy_data_by_8_fold(node_xy)
            node_prize      = node_prize.repeat(8, 1)
            selected_indices = selected_indices.repeat(8, 1)

        # ── (x, y, prize) node features ──────────────────────────────
        node_xy_prize = torch.cat(
            [node_xy, node_prize[:, :, None]], dim=2
        )  # (batch, top_k, 3)

        self.node_xy          = node_xy
        self.node_prize       = node_prize
        self.node_xy_prize    = node_xy_prize
        self.selected_indices = selected_indices

        self.BATCH_IDX = torch.arange(self.batch_size, device=device)[:, None] \
                              .expand(self.batch_size, self.pomo_size)
        self.POMO_IDX  = torch.arange(self.pomo_size,  device=device)[None, :] \
                              .expand(self.batch_size, self.pomo_size)

        # ── Populate reset_state ──────────────────────────────────────
        self.reset_state.node_xy          = node_xy
        self.reset_state.node_prize       = node_prize
        self.reset_state.node_xy_prize    = node_xy_prize
        self.reset_state.selected_scores  = node_prize          # alias
        self.reset_state.selected_indices = selected_indices

        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.POMO_IDX  = self.POMO_IDX

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------
    def reset(self):
        device = self.node_xy.device
        B, P, K = self.batch_size, self.pomo_size, self.problem_size

        self.selected_count     = 0
        self.current_node       = None
        self.selected_node_list = torch.zeros((B, P, 0), dtype=torch.long, device=device)

        self.collected_prize    = torch.zeros(B, P, device=device)
        self.remaining_budget   = torch.full((B, P), self.budget, device=device)

        self.visited_mask       = torch.zeros(B, P, K, dtype=torch.bool, device=device)
        self.ninf_mask          = torch.zeros(B, P, K, device=device)
        self.finished           = torch.zeros(B, P, dtype=torch.bool, device=device)

        # start_xy is set at first step when current_node is chosen
        self.start_xy           = None

        return self.reset_state, None, False

    # ------------------------------------------------------------------
    # pre_step
    # ------------------------------------------------------------------
    def pre_step(self):
        self.step_state.current_node     = self.current_node
        self.step_state.ninf_mask        = self.ninf_mask
        self.step_state.remaining_budget = self.remaining_budget
        self.step_state.finished         = self.finished
        return self.step_state, None, False

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------
    def step(self, selected):
        """
        Args:
            selected: (batch, pomo)  -- index into top_k nodes [0, top_k)

        Returns:
            step_state, reward (None until done), done (bool)
        """
        B, P = self.batch_size, self.pomo_size
        device = self.node_xy.device

        self.selected_count += 1
        self.current_node    = selected
        self.selected_node_list = torch.cat(
            [self.selected_node_list, selected[:, :, None]], dim=2
        )

        # ── Record start position for each pomo trajectory ────────────
        # On the very first step, start_xy = position of chosen start node.
        if self.start_xy is None:
            # selected: (B, P) — gather (x,y) for each
            gather_idx = selected[:, :, None, None].expand(B, P, 1, 2)
            node_xy_exp = self.node_xy[:, None, :, :].expand(B, P, -1, 2)
            self.start_xy = node_xy_exp.gather(dim=2, index=gather_idx).squeeze(2)
            # shape: (B, P, 2)

        # ── Collect prize ─────────────────────────────────────────────
        prize_list = self.node_prize[:, None, :].expand(B, P, -1)
        selected_prize = prize_list.gather(dim=2, index=selected[:, :, None]).squeeze(2)
        self.collected_prize += selected_prize * (~self.finished).float()

        # ── Update travel budget ───────────────────────────────────────
        # Subtract dist(prev → current).
        # On step 1 there is no previous, so dist = 0.
        if self.selected_count > 1:
            prev_node = self.selected_node_list[:, :, -2]   # node before current
            gather_prev = prev_node[:, :, None, None].expand(B, P, 1, 2)
            gather_curr = selected[:, :, None, None].expand(B, P, 1, 2)
            node_xy_exp = self.node_xy[:, None, :, :].expand(B, P, -1, 2)
            prev_xy = node_xy_exp.gather(dim=2, index=gather_prev).squeeze(2)  # (B,P,2)
            curr_xy = node_xy_exp.gather(dim=2, index=gather_curr).squeeze(2)  # (B,P,2)
            step_dist = ((curr_xy - prev_xy) ** 2).sum(dim=2).sqrt()           # (B,P)
            self.remaining_budget -= step_dist * (~self.finished).float()

        # ── Mark visited ──────────────────────────────────────────────
        self.visited_mask[self.BATCH_IDX, self.POMO_IDX, selected] = True

        # ── Recompute ninf_mask ───────────────────────────────────────
        # A node is infeasible if:
        #   (a) already visited, OR
        #   (b) dist(current → node) + dist(node → start) > remaining_budget
        sel_idx  = selected[:, :, None, None].expand(B, P, 1, 2)   # (B,P,1,2)
        node_exp = self.node_xy[:, None, :, :].expand(B, P, self.problem_size, 2)
        curr_xy  = node_exp.gather(dim=2, index=sel_idx).squeeze(2)  # (B,P,2)

        # dist from current node to every candidate
        dist_to_cand = (
            (node_exp - curr_xy[:, :, None, :]) ** 2
        ).sum(dim=3).sqrt()   # (B, P, K)

        # dist from every candidate back to start
        start_exp    = self.start_xy[:, :, None, :].expand(B, P, self.problem_size, 2)
        dist_to_start = (
            (node_exp - start_exp) ** 2
        ).sum(dim=3).sqrt()   # (B, P, K)

        # remaining budget per pomo, expanded
        budget_exp = self.remaining_budget[:, :, None].expand(B, P, self.problem_size)

        budget_infeasible = (dist_to_cand + dist_to_start) > budget_exp

        self.ninf_mask = torch.zeros(B, P, self.problem_size, device=device)
        self.ninf_mask[self.visited_mask]    = float('-inf')
        self.ninf_mask[budget_infeasible]    = float('-inf')

        # ── Check done ────────────────────────────────────────────────
        # A trajectory is finished when all nodes are either visited or infeasible
        all_masked = (self.ninf_mask == float('-inf')).all(dim=2)   # (B, P)
        self.finished = self.finished | all_masked

        # ── Update step_state ─────────────────────────────────────────
        self.step_state.current_node     = self.current_node
        self.step_state.ninf_mask        = self.ninf_mask
        self.step_state.remaining_budget = self.remaining_budget
        self.step_state.finished         = self.finished

        done   = self.finished.all()
        # Reward = total prize collected (positive).  Emitted only at end.
        reward = self.collected_prize if done else None

        return self.step_state, reward, done
