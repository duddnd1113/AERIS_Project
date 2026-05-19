import os
import logging
import torch

from rl.OPTopKEnvironment import OPTopKEnv as Env
from rl.OPTopKModel import OPTopKModel as Model


def build_params(total_grid_size, top_k, budget, lr, weight_decay):
    env_params = {
        "total_grid_size": total_grid_size,
        "top_k": top_k,
        "pomo_size": top_k,
        "budget": budget,
    }

    model_params = {
        "embedding_dim": 128,
        "sqrt_embedding_dim": 128 ** 0.5,
        "encoder_layer_num": 6,
        "qkv_dim": 16,
        "head_num": 8,
        "logit_clipping": 10,
        "ff_hidden_dim": 512,
        "eval_type": "argmax",
    }

    optimizer_params = {
        "lr": lr,
        "weight_decay": weight_decay,
    }

    return env_params, model_params, optimizer_params


def setup_logger(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    logger = logging.getLogger("active_search")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    stream_handler = logging.StreamHandler()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def train_one_snapshot(
    model,
    env,
    optimizer,
    coords,
    scores,
    epochs,
    log_interval,
    logger,
    checkpoint_path=None,
    sim_id=None
):
    model.train()

    best_prize = -1.0

    for epoch in range(1, epochs + 1):
        env.load_problems(
            batch_size=1,
            aug_factor=1,
            coords=coords,
            scores=scores,
        )

        reset_state, _, _ = env.reset()
        model.pre_forward(reset_state)

        prob_list = torch.zeros(
            size=(1, env.pomo_size, 0),
            device=coords.device
        )

        state, reward, done = env.pre_step()

        while not done:
            selected, prob = model(state)
            state, reward, done = env.step(selected)

            if state.finished is not None:
                prob = prob.masked_fill(state.finished, 1.0)

            prob_list = torch.cat(
                (prob_list, prob[:, :, None]),
                dim=2
            )

        advantage = reward - reward.float().mean(dim=1, keepdim=True)
        log_prob = prob_list.log().sum(dim=2)
        loss = -(advantage * log_prob).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_best = reward.max(dim=1).values.mean().item()
        epoch_mean = reward.float().mean().item()

        if epoch_best > best_prize:
            best_prize = epoch_best

        if checkpoint_path is not None:
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_prize": best_prize,
                    "sim_id": sim_id,
                },
                checkpoint_path,
            )

        if epoch == 1 or epoch % log_interval == 0 or epoch == epochs:
            logger.info(
                f"sim={sim_id} | epoch={epoch}/{epochs} | "
                f"best_prize={epoch_best:.4f} | "
                f"mean_prize={epoch_mean:.4f} | "
                f"global_best={best_prize:.4f} | "
                f"loss={loss.item():.6f}"
            )

    return model


def test_one_snapshot(model, env, coords, scores):
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

    flat_best = reward.reshape(-1).argmax().item()
    best_aug = flat_best // env.pomo_size
    best_pomo = flat_best % env.pomo_size

    best_prize = reward[best_aug, best_pomo].item()

    local_route = env.selected_node_list[
        best_aug,
        best_pomo
    ].cpu().tolist()

    topk_original_indices = env.selected_indices[0].cpu().tolist()

    route_original_indices = [
        topk_original_indices[j]
        for j in local_route
    ]

    return route_original_indices, best_prize


def solve_active_search(
    df,
    budget=3.0,
    epochs=500,
    lr=1e-4,
    weight_decay=1e-6,
    top_k=None,
    device="cpu",
    log_interval=10,
    sim_id=None,
    checkpoint_dir="outputs/checkpoints"
):
    logger = setup_logger("outputs/logs/active_search.log")

    if top_k is None:
        top_k = len(df)

    total_grid_size = len(df)

    coords = torch.tensor(
        df[["x", "y"]].values,
        dtype=torch.float32,
        device=device
    ).unsqueeze(0)

    scores = torch.tensor(
        df["score"].values,
        dtype=torch.float32,
        device=device
    ).unsqueeze(0)

    env_params, model_params, optimizer_params = build_params(
        total_grid_size=total_grid_size,
        top_k=top_k,
        budget=budget,
        lr=lr,
        weight_decay=weight_decay,
    )

    env = Env(**env_params)
    model = Model(**model_params).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        **optimizer_params
    )

    checkpoint_path = os.path.join(
        checkpoint_dir,
        f"active_search_sim_{sim_id}.pt"
    )

    train_one_snapshot(
        model=model,
        env=env,
        optimizer=optimizer,
        coords=coords,
        scores=scores,
        epochs=epochs,
        log_interval=log_interval,
        logger=logger,
        checkpoint_path=checkpoint_path,
        sim_id=sim_id,
    )

    route, best_prize = test_one_snapshot(
        model=model,
        env=env,
        coords=coords,
        scores=scores,
    )

    logger.info(
        f"sim={sim_id} | TEST best_prize={best_prize:.4f} | "
        f"nodes={len(route)}"
    )

    return route

