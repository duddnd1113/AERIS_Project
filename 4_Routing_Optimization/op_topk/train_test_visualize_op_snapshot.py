"""
train_test_visualize_op_snapshot.py
=====================================
Active Search on ONE scored grid snapshot for the Orienteering Problem (OP).

Flow:
    1. Generate (or load) 18K grid: (x, y, score) per cell
    2. Select top-300 nodes by score as candidates
    3. Normalize scores → prizes in [0, 1]
    4. Active Search: train POMO-OP on this fixed snapshot
          - Each epoch: run top_k parallel POMO trajectories
          - Reward = total prize collected within budget T
          - POMO shared baseline: mean reward across all starts
          - Gradient pushes policy toward higher-prize routes
    5. Test: best route from all POMO starts + 8x augmentation
    6. Save route CSV and visualization

Key parameters:
    --top-k 300        candidate nodes fed to the model
    --budget 3.0       max travel distance (normalized coords 0–1)
    --epochs 500       more epochs = better Active Search solution

Example:
    python train_test_visualize_op_snapshot.py \
        --total-grid-size 18225 \
        --top-k 300 \
        --budget 3.0 \
        --epochs 500 \
        --output-fig op_route.png
"""

import argparse
import logging

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from OPTopKEnvironment import OPTopKEnv as Env
from OPTopKModel import OPTopKModel as Model


# ============================================================
# 1. Hyperparameters
# ============================================================

def build_params(args):
    env_params = {
        'total_grid_size': args.total_grid_size,
        'top_k':           args.top_k,
        'pomo_size':       args.top_k,
        'budget':          args.budget,
    }
    model_params = {
        'embedding_dim':       128,
        'sqrt_embedding_dim':  128 ** 0.5,
        'encoder_layer_num':   6,
        'qkv_dim':             16,
        'head_num':            8,
        'logit_clipping':      10,
        'ff_hidden_dim':       512,
        'eval_type':           'argmax',
    }
    optimizer_params = {
        'lr':           args.lr,
        'weight_decay': args.weight_decay,
    }
    return env_params, model_params, optimizer_params


# ============================================================
# 2. Snapshot generation
# ============================================================

def generate_synthetic_snapshot(args, device):
    """Create a synthetic 18K scored grid for demonstration.

    Replace this function with your real (x, y, score) data at inference.
    The only requirement: coords (1, N, 2) and scores (1, N) as tensors.
    """
    torch.manual_seed(args.seed)

    N = args.total_grid_size
    grid_size = int(N ** 0.5)
    if grid_size * grid_size != N:
        raise ValueError(
            f"total_grid_size must be a perfect square. "
            f"Try 18225 (135²) instead of {N}."
        )

    x = torch.linspace(0, 1, grid_size, device=device)
    y = torch.linspace(0, 1, grid_size, device=device)
    try:
        xv, yv = torch.meshgrid(x, y, indexing='ij')
    except TypeError:
        xv, yv = torch.meshgrid(x, y)

    coords = torch.stack([xv.reshape(-1), yv.reshape(-1)], dim=1).unsqueeze(0)  # (1, N, 2)
    scores = torch.rand(1, N, device=device)   # uniform random — replace with real scores

    df = pd.DataFrame({
        'grid_id': range(N),
        'x':       coords[0, :, 0].cpu().numpy(),
        'y':       coords[0, :, 1].cpu().numpy(),
        'score':   scores[0].cpu().numpy(),
    })

    logging.info(
        'Grid snapshot | N=%d | score min=%.4f max=%.4f mean=%.4f',
        N, scores.min(), scores.max(), scores.mean(),
    )
    return df, coords, scores


# ============================================================
# 3. Active Search training
# ============================================================

def train_one_snapshot(model, env, optimizer, coords, scores, args):
    """
    Active Search: overfit the policy to this one fixed instance.

    Each epoch:
        - Run top_k POMO trajectories in parallel from different start nodes
        - Reward = sum of prizes collected within budget (positive number)
        - Shared POMO baseline = mean reward across all starts
        - advantage_i = R_i - mean(R)  → reinforce better-than-average starts
        - Gradient update pushes policy toward higher-prize routes

    The model learns:
        - Which nodes are worth visiting given their (x, y, prize)
        - When to stop (budget constraint via ninf_mask in environment)
        - Efficient orderings that maximize prize within distance budget
    """
    model.train()

    for epoch in range(1, args.epochs + 1):
        env.load_problems(
            batch_size=1,
            aug_factor=1,
            coords=coords,
            scores=scores,
        )

        reset_state, _, _ = env.reset()
        model.pre_forward(reset_state)

        prob_list = torch.zeros(
            size=(1, env.pomo_size, 0), device=coords.device
        )

        state, reward, done = env.pre_step()

        while not done:
            selected, prob = model(state)
            state, reward, done = env.step(selected)

            # Finished trajectories must not contribute to the loss.
            # Setting prob=1 makes log(prob)=0, so their gradient is zero.
            if state.finished is not None:
                prob = prob.masked_fill(state.finished, 1.0)

            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

        # reward: (1, pomo_size)  — total prize per trajectory
        # POMO shared baseline: mean prize across all starts
        advantage  = reward - reward.float().mean(dim=1, keepdim=True)
        log_prob   = prob_list.log().sum(dim=2)
        loss       = -(advantage * log_prob).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % args.log_interval == 0 or epoch == args.epochs:
            best_prize = reward.max(dim=1).values.mean().item()
            mean_prize = reward.float().mean().item()
            logging.info(
                'Epoch %d/%d | best_prize=%.4f | mean_prize=%.4f | loss=%.6f',
                epoch, args.epochs, best_prize, mean_prize, loss.item(),
            )

    return model


# ============================================================
# 4. Test
# ============================================================

def test_one_snapshot(model, env, coords, scores):
    """
    Inference with 8x coordinate augmentation.
    8 augmentations × top_k POMO starts = 8 × 300 = 2400 candidate routes.
    Take the single best.
    """
    model.eval()

    with torch.no_grad():
        env.load_problems(
            batch_size=1,
            aug_factor=8,
            coords=coords,
            scores=scores,
        )

        reset_state, _, _ = env.reset()
        model.pre_forward(reset_state)

        state, reward, done = env.pre_step()
        while not done:
            selected, _ = model(state)
            state, reward, done = env.step(selected)

    # reward: (8, pomo_size) — prize per trajectory per augmentation
    flat_best = reward.reshape(-1).argmax().item()
    best_aug  = flat_best // env.pomo_size
    best_pomo = flat_best % env.pomo_size

    best_prize = reward[best_aug, best_pomo].item()

    local_route          = env.selected_node_list[best_aug, best_pomo].cpu().tolist()
    topk_original_indices = env.selected_indices[0].cpu().tolist()   # same for all augs
    route_original_indices = [topk_original_indices[j] for j in local_route]

    return {
        'best_aug':               best_aug,
        'best_pomo':              best_pomo,
        'best_prize':             best_prize,
        'local_route':            local_route,
        'topk_original_indices':  topk_original_indices,
        'route_original_indices': route_original_indices,
    }


# ============================================================
# 5. Save CSV
# ============================================================

def save_route_csv(df, route_indices, filename):
    route_df = df.iloc[route_indices].copy()
    route_df.insert(0, 'route_order', range(len(route_df)))
    route_df.to_csv(filename, index=False)
    logging.info('Saved route CSV: %s', filename)
    return route_df


# ============================================================
# 6. Visualize
# ============================================================

def visualize(df, result, args):
    x    = df['x'].values
    y    = df['y'].values
    s    = df['score'].values

    topk_idx  = result['topk_original_indices']
    route_idx = result['route_original_indices']
    prize     = result['best_prize']

    OFF_BLACK   = '#111827'
    COOL_GRAY   = '#F8FAFC'
    PANEL       = '#F1F5F9'
    GRID_LINE   = '#CBD5E1'
    SAT_BLUE    = '#4F46E5'
    LIGHT_BLUE  = '#DBEAFE'

    blue_cmap = LinearSegmentedColormap.from_list(
        'op_blue', [PANEL, LIGHT_BLUE, SAT_BLUE]
    )

    fig, ax = plt.subplots(figsize=(11, 9), facecolor=COOL_GRAY)
    ax.set_facecolor(COOL_GRAY)

    # Full grid — colored by score
    sc = ax.scatter(
        x, y, c=s, cmap=blue_cmap,
        s=args.grid_marker_size, alpha=0.7, edgecolors='none',
    )
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('score / prize', color=OFF_BLACK)
    cbar.ax.tick_params(colors=OFF_BLACK)
    cbar.outline.set_edgecolor(GRID_LINE)

    # Top-k candidates
    ax.scatter(
        x[topk_idx], y[topk_idx],
        s=args.topk_marker_size, marker='o',
        facecolors='none', edgecolors=OFF_BLACK,
        linewidths=0.8, alpha=0.5,
        label=f'Top-{args.top_k} candidates',
    )

    # Learned route
    rx, ry = x[route_idx], y[route_idx]
    ax.plot(rx, ry, color=SAT_BLUE, linewidth=2.5, zorder=4, label='OP route')

    if len(route_idx) > 1:
        ax.plot(
            [rx[-1], rx[0]], [ry[-1], ry[0]],
            color=OFF_BLACK, linewidth=1.5, linestyle='--',
            alpha=0.6, zorder=3, label='Return',
        )

    # Start node
    ax.scatter(
        rx[0], ry[0],
        s=args.start_marker_size, marker='*',
        color=SAT_BLUE, edgecolors=OFF_BLACK, linewidths=1.0,
        zorder=5, label='Start',
    )

    ax.set_title(
        f'OP Active Search  |  prize = {prize:.4f}  |  '
        f'nodes visited = {len(route_idx)}  |  budget = {args.budget}',
        color=OFF_BLACK, fontweight='bold',
    )
    ax.set_xlabel('x', color=OFF_BLACK)
    ax.set_ylabel('y', color=OFF_BLACK)
    ax.tick_params(colors=OFF_BLACK)
    ax.grid(True, color=GRID_LINE, alpha=0.4, linewidth=0.7)
    for sp in ax.spines.values():
        sp.set_color(GRID_LINE)
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(args.output_fig, dpi=300, facecolor=fig.get_facecolor())
    plt.close(fig)
    logging.info('Saved figure: %s', args.output_fig)


# ============================================================
# 7. Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Active Search POMO-OP on one scored grid snapshot.'
    )

    # Grid
    parser.add_argument('--total-grid-size', type=int, default=18225,
                        help='Must be a perfect square. 18225 = 135².')
    parser.add_argument('--top-k', type=int, default=300,
                        help='Candidate nodes selected by score.')

    # OP
    parser.add_argument('--budget', type=float, default=3.0,
                        help='Max travel distance (coords normalized 0–1).')

    # Training
    parser.add_argument('--epochs',       type=int,   default=500)
    parser.add_argument('--lr',           type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-6)
    parser.add_argument('--log-interval', type=int,   default=10)
    parser.add_argument('--seed',         type=int,   default=42)

    # Output
    parser.add_argument('--checkpoint',   default='checkpoint-op-snapshot.pt')
    parser.add_argument('--output-route', default='op_route_output.csv')
    parser.add_argument('--output-fig',   default='op_route_visualization.png')

    # Hardware
    parser.add_argument('--use-cuda',        action='store_true')
    parser.add_argument('--cuda-device-num', type=int, default=0)

    # Viz
    parser.add_argument('--grid-marker-size',  type=float, default=4.0)
    parser.add_argument('--topk-marker-size',  type=float, default=40.0)
    parser.add_argument('--start-marker-size', type=float, default=160.0)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
    )

    # ── Device ──────────────────────────────────────────────
    use_cuda = args.use_cuda and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(args.cuda_device_num)
        device = torch.device('cuda', args.cuda_device_num)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        logging.info('Using CUDA device %d', args.cuda_device_num)
    else:
        device = torch.device('cpu')
        torch.set_default_tensor_type('torch.FloatTensor')
        logging.info('Using CPU')

    if args.top_k > args.total_grid_size:
        raise ValueError(f'top_k={args.top_k} > total_grid_size={args.total_grid_size}')

    # ── Build ────────────────────────────────────────────────
    env_params, model_params, optimizer_params = build_params(args)

    df, coords, scores = generate_synthetic_snapshot(args, device)

    env       = Env(**env_params)
    model     = Model(**model_params).to(device)
    optimizer = torch.optim.Adam(model.parameters(), **optimizer_params)

    logging.info(
        'Active Search OP | grid=%d | top_k=%d | budget=%.2f | epochs=%d',
        args.total_grid_size, args.top_k, args.budget, args.epochs,
    )

    # ── Train ────────────────────────────────────────────────
    train_one_snapshot(model, env, optimizer, coords, scores, args)

    torch.save({
        'model_state_dict': model.state_dict(),
        'env_params':       env_params,
        'model_params':     model_params,
        'args':             vars(args),
    }, args.checkpoint)
    logging.info('Checkpoint saved: %s', args.checkpoint)

    # ── Test ─────────────────────────────────────────────────
    result = test_one_snapshot(model, env, coords, scores)

    route_idx = result['route_original_indices']
    prize     = result['best_prize']

    logging.info(
        'Best route | pomo=%d aug=%d | prize=%.4f | nodes=%d',
        result['best_pomo'], result['best_aug'], prize, len(route_idx),
    )

    save_route_csv(df, route_idx, args.output_route)
    visualize(df, result, args)

    print('\n' + '=' * 55)
    print(f'  prize collected : {prize:.4f}')
    print(f'  nodes visited   : {len(route_idx)} / {args.top_k}')
    print(f'  budget          : {args.budget}')
    print(f'  checkpoint      : {args.checkpoint}')
    print(f'  route CSV       : {args.output_route}')
    print(f'  figure          : {args.output_fig}')
    print('=' * 55)


if __name__ == '__main__':
    main()
