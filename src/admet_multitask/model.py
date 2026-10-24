"""Matched MLPs, task-balanced masked training, validation-only stopping."""
from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Network(nn.Module):
    def __init__(self, input_dim=1024, hidden=(128, 64), n_tasks=3):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(input_dim, hidden[0]), nn.ReLU(),
                                   nn.Linear(hidden[0], hidden[1]), nn.ReLU())
        self.heads = nn.ModuleList([nn.Linear(hidden[1], 1) for _ in range(n_tasks)])

    def forward(self, x, task):
        return self.heads[task](self.trunk(x)).squeeze(-1)


def observed_loss(pred, target, classification=False):
    mask = torch.isfinite(target)
    if not mask.any():
        return pred.sum() * 0
    if classification:
        return F.binary_cross_entropy_with_logits(pred[mask], target[mask])
    return F.mse_loss(pred[mask], target[mask])


def fit(x, y, split, tasks, seed, config):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = Network(input_dim=x.shape[1], hidden=tuple(config["hidden"]), n_tasks=y.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"],
                                 weight_decay=config["weight_decay"])
    train = {t: np.flatnonzero((split == "train") & np.isfinite(y[:, t])) for t in tasks}
    valid = {t: np.flatnonzero((split == "valid") & np.isfinite(y[:, t])) for t in tasks}
    # Global schedule gives every task exactly one pass per epoch, for both methods.
    steps = int(np.ceil(max(np.sum((split == "train") & np.isfinite(y[:, t]))
                            for t in range(y.shape[1])) / config["batch_size"]))
    if min(map(len, train.values())) < steps:
        raise ValueError("Too few labels for task-balanced batches")
    xt = torch.from_numpy(x)
    yt = torch.from_numpy(y.astype(np.float32))
    best, best_epoch, stale = float("inf"), 0, 0
    state = deepcopy(model.state_dict())
    history = []
    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        # Task-specific RNG makes batches identical between ST and MTL.
        batches = {t: np.array_split(np.random.default_rng(seed * 100000 + epoch * 10 + t)
                                    .permutation(train[t]), steps) for t in tasks}
        for step in range(steps):
            optimizer.zero_grad()
            losses = [observed_loss(model(xt[batches[t][step]], t), yt[batches[t][step], t], t == 2)
                      for t in tasks]
            loss = torch.stack(losses).mean()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            score = float(torch.stack([observed_loss(model(xt[valid[t]], t), yt[valid[t], t], t == 2)
                                       for t in tasks]).mean())
        history.append({"epoch": epoch, "validation_loss": score})
        if score < best - 1e-6:
            best, best_epoch, stale = score, epoch, 0
            state = deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= config["patience"]:
            break
    model.load_state_dict(state)
    model.eval()
    return model, {"best_epoch": best_epoch, "epochs_run": epoch, "validation_loss": best,
                   "history": history}
