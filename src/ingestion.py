import os

from sklearn.datasets import load_breast_cancer

os.makedirs("data/raw", exist_ok=True)

data = load_breast_cancer(as_frame=True)
df = data.frame
df.to_csv("data/raw/data.csv", index=False)

print("✅ Raw data written")
