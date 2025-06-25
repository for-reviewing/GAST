import os
import json
import pandas as pd

import sys
from pathlib import Path

# Dynamically add the project root to PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(PROJECT_ROOT))

print(f"Project root: {PROJECT_ROOT}")

from src.data.data_loader import DATASET_NAME_LIST
# All seeds to be checked
seeds = list(range(42, 352, 10))

results = []

for dname in DATASET_NAME_LIST:
    for seed in seeds:
        json_path = f"{PROJECT_ROOT}/models/final/gast/{dname}/test_results/metrics_seed_{seed}.json"
        if os.path.exists(json_path):
            try:
                with open(json_path, "r") as f:
                    metrics = json.load(f)
                oa = metrics.get("Overall Accuracy", None)
                if oa is not None:
                    results.append({
                        "Dataset": dname,
                        "OA": oa,
                        "seed": seed
                    })
            except Exception as e:
                print(f"Error reading {json_path}: {e}")


df = pd.DataFrame(results)

df = df.sort_values(["Dataset", "OA"], ascending=[True, False])

df.to_csv("all_results_sorted.csv", index=False)

print(df)
print(f"Results saved to path: {os.path.abspath('all_results_sorted.csv')}")