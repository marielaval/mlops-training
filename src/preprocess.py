# src/preprocess.py
import os

import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/raw/data.csv")

X = df.drop(columns=["target"])
y = df["target"]

X_train, X_tmp, y_train, y_tmp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42
)

os.makedirs("data/processed", exist_ok=True)

for name, X_, y_ in [
    ("train", X_train, y_train),
    ("val", X_val, y_val),
    ("test", X_test, y_test),
]:
    X_.to_csv(f"data/processed/X_{name}.csv", index=False)
    y_.to_csv(f"data/processed/y_{name}.csv", index=False)

print("✅ Data processed")
