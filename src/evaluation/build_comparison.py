# src/evaluation/build_comparison.py
"""
Собирает финальную таблицу сравнения всех моделей
из логов обучения (runs/logs/*.json) и метрик PatchCore.
"""
import json
import os
import pandas as pd

LOGS_DIR = "runs/logs"
METRICS_DIR = "runs/metrics"
OUTPUT_CSV = "runs/final_model_comparison.csv"

rows = []

# ── Классификаторы (из логов обучения) ──
for fname in os.listdir(LOGS_DIR):
    if not fname.endswith(".json"):
        continue

    model_name = fname.replace(".json", "")
    with open(os.path.join(LOGS_DIR, fname)) as f:
        log = json.load(f)

    history = log.get("history", [])
    if not history:
        continue

    # эпоха с лучшим val_f1
    best_epoch = max(history, key=lambda h: h["val_f1"])

    rows.append({
        "model": model_name,
        "type": "classifier",
        "accuracy": round(best_epoch["val_acc"], 4),
        "precision": round(best_epoch["val_precision"], 4),
        "recall": round(best_epoch["val_recall"], 4),
        "f1": round(best_epoch["val_f1"], 4),
        "best_epoch": best_epoch["epoch"],
        "total_epochs": len(history),
    })

# ── PatchCore (из patchcore_stats.pt / patchcore_eval.json) ──
patchcore_eval_path = os.path.join(METRICS_DIR, "patchcore_eval.json")
if os.path.exists(patchcore_eval_path):
    with open(patchcore_eval_path) as f:
        pc_eval = json.load(f)

    rows.append({
        "model": "patchcore",
        "type": "anomaly_detection",
        "accuracy": round(pc_eval.get("accuracy", float("nan")), 4) if pc_eval.get("accuracy") else None,
        "precision": round(pc_eval.get("precision", float("nan")), 4) if pc_eval.get("precision") else None,
        "recall": round(pc_eval.get("recall", float("nan")), 4) if pc_eval.get("recall") else None,
        "f1": round(pc_eval.get("f1", float("nan")), 4) if pc_eval.get("f1") else None,
        "best_epoch": None,
        "total_epochs": None,
    })

# ── Save ──
df = pd.DataFrame(rows)
df = df.sort_values("f1", ascending=False).reset_index(drop=True)

os.makedirs("runs", exist_ok=True)
df.to_csv(OUTPUT_CSV, index=False)

print("Final model comparison:")
print(df.to_string(index=False))
print(f"\nSaved to {OUTPUT_CSV}")