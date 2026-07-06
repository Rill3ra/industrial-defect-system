# src/evaluation/final_comparison.py

import os
import json
import pandas as pd


METRICS_DIR = "runs/metrics"


rows = []

for filename in os.listdir(METRICS_DIR):

    if not filename.endswith(".json"):
        continue

    if filename == "patchcore_eval.json":

        with open(os.path.join(METRICS_DIR, filename), "r") as f:
            data = json.load(f)

        rows.append({
            "model": "patchcore",
            "accuracy": data.get("accuracy", 0.0),
            "precision": data.get("precision", 0.0),
            "recall": data.get("recall", 0.0),
            "f1": data.get("f1", 0.0),
            "roc_auc": data.get("roc_auc", 0.0)
        })

    else:

        with open(
            os.path.join(METRICS_DIR, filename),
            "r"
        ) as f:

            data = json.load(f)

        rows.append({
            "model": data["model"],
            "accuracy": data["accuracy"],
            "precision": data["precision"],
            "recall": data["recall"],
            "f1": data["f1"],
            "roc_auc": data["roc_auc"]
        })


df = pd.DataFrame(rows)

df = df.sort_values(
    by="f1",
    ascending=False
)

os.makedirs(
    "runs",
    exist_ok=True
)

output_path = "runs/final_model_comparison.csv"

df.to_csv(
    output_path,
    index=False
)

print("\n=== FINAL MODEL COMPARISON ===\n")

print(
    df.to_string(index=False)
)

print("\nSaved:", output_path)

print("\nBEST MODEL:")
print(df.iloc[0]["model"])

print(
    "BEST F1:",
    round(df.iloc[0]["f1"], 4)
)