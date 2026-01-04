# serving/app.py
import pandas as pd
import torch
from fastapi import FastAPI
from serving.model import MLP

app = FastAPI()

FEATURES = pd.read_csv("data/processed/X_train.csv").columns.tolist()

state_dict = torch.load("models/current/model.pt", map_location="cpu")
fixed_state_dict = {}
for k, v in state_dict.items():
    fixed_state_dict[f"net.{k}"] = v
model = MLP(len(FEATURES))
model.load_state_dict(fixed_state_dict)
model.eval()


@app.get("/")
def root():
    return {"message": "API is running"}


@app.post("/predict")
def predict(features: dict):
    x = torch.tensor([[features[f] for f in FEATURES]], dtype=torch.float32)
    with torch.no_grad():
        proba = model(x).item()
    return {"proba": proba, "pred": int(proba > 0.5)}
