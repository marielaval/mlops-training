import itertools
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MLP(nn.Module):
    def __init__(self, n_features: int, hidden_size: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def load_data(batch_size: int):
    X_train = pd.read_csv("data/processed/X_train.csv").values
    y_train = pd.read_csv("data/processed/y_train.csv").values
    X_val = pd.read_csv("data/processed/X_val.csv").values
    y_val = pd.read_csv("data/processed/y_val.csv").values

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1)

    train_dl = DataLoader(
        TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True
    )
    return X_train, y_train, X_val, y_val, train_dl


def train_one(cfg, hparams):
    epochs = cfg["experiment"]["epochs"]
    batch_size = cfg["experiment"]["batch_size"]

    X_train, y_train, X_val, y_val, train_dl = load_data(batch_size)
    n_features = X_train.shape[1]

    model = MLP(
        n_features=n_features,
        hidden_size=hparams["hidden_size"],
        dropout=hparams["dropout"],
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hparams["lr"],
        weight_decay=hparams["weight_decay"],
    )
    loss_fn = nn.BCELoss()

    for _ in range(epochs):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        val_pred = model(X_val.to(DEVICE))
        val_acc = ((val_pred > 0.5) == y_val.to(DEVICE)).float().mean().item()

    return model, {"val_accuracy": float(val_acc)}


def main():
    with open("src/config.yml", "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["experiment"].get("seed", 42))

    # Build grid
    grid = {
        "hidden_size": cfg["model"]["hidden_size"],
        "dropout": cfg["model"]["dropout"],
        "lr": cfg["optim"]["lr"],
        "weight_decay": cfg["optim"]["weight_decay"],
    }
    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))

    metric = cfg["experiment"]["metric"]
    maximize = bool(cfg["experiment"]["maximize"])

    best_score = -1e18 if maximize else 1e18
    best = None

    os.makedirs("models/registry", exist_ok=True)

    for i, values in enumerate(combos, 1):
        hparams = dict(zip(keys, values))
        model, metrics = train_one(cfg, hparams)

        score = metrics[metric]
        is_better = score > best_score if maximize else score < best_score

        # log each run (utile pour comparer)
        run_id = time.strftime("%Y%m%d-%H%M%S") + f"-run{i:03d}"
        run_dir = f"models/registry/{run_id}"
        os.makedirs(run_dir, exist_ok=True)
        torch.save(model.state_dict(), f"{run_dir}/model.pt")
        json.dump(
            {"hparams": hparams, **metrics},
            open(f"{run_dir}/metrics.json", "w"),
            indent=2,
        )

        print(f"[{i}/{len(combos)}] {hparams} -> {metrics}")

        if is_better:
            best_score = score
            best = {"run_id": run_id, "hparams": hparams, "metrics": metrics}

    # Résumé best
    json.dump(best, open("models/registry/best.json", "w"), indent=2)
    print("✅ Best:", best)


if __name__ == "__main__":
    main()
