import json
import os
import time

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

X_train = pd.read_csv("data/processed/X_train.csv").values
y_train = pd.read_csv("data/processed/y_train.csv").values
X_val = pd.read_csv("data/processed/X_val.csv").values
y_val = pd.read_csv("data/processed/y_val.csv").values

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1)

train_ds = TensorDataset(X_train, y_train)
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)

model = nn.Sequential(
    nn.Linear(X_train.shape[1], 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid()
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

loss_fn = nn.BCELoss()

for epoch in range(20):
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

version = time.strftime("%Y%m%d-%H%M%S")
out = f"models/registry/{version}"
os.makedirs(out, exist_ok=True)

torch.save(model.state_dict(), f"{out}/model.pt")
json.dump({"val_accuracy": val_acc}, open(f"{out}/metrics.json", "w"))

print("✅ Model trained", val_acc)
