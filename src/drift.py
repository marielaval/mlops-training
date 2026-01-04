# src/drift.py
import pandas as pd
from scipy.stats import ks_2samp

train = pd.read_csv("data/processed/X_train.csv")
live = pd.read_csv("data/live/live.csv")

for col in train.columns:
    stat, p = ks_2samp(train[col], live[col])
    if p < 0.05:
        print(f"⚠️ Drift detected on {col}")
