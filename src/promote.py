# src/promote.py
import os
import shutil
import sys

version = sys.argv[1]
src = f"models/registry/{version}"
dst = "models/current"

os.makedirs(dst, exist_ok=True)

shutil.copy(f"{src}/model.pt", f"{dst}/model.pt")
shutil.copy(f"{src}/metrics.json", f"{dst}/metrics.json")

print("🚀 Model promoted")
