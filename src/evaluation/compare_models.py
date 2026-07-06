import os
import json
import pandas as pd


LOG_DIR = "runs/logs"


def load_logs():
    rows = []

    for file in os.listdir(LOG_DIR):
        if not file.endswith(".json"):
            continue

        path = os.path.join(LOG_DIR, file)

        with open(path, "r") as f:
            data = json.load(f)

        rows.append({
            "model": data.get("model"),
            "best_f1": data.get("best_f1", 0)
        })

    return pd.DataFrame(rows)


def main():

    df = load_logs()

    if df.empty:
        print("No logs found.")
        return

    df = df.sort_values(by="best_f1", ascending=False)

    print("\n=== MODEL COMPARISON ===\n")
    print(df.to_string(index=False))

    best = df.iloc[0]

    print("\n========================")
    print(f"BEST MODEL: {best['model']}")
    print(f"BEST F1: {best['best_f1']:.4f}")
    print("========================\n")

    os.makedirs("runs/metrics", exist_ok=True)

    df.to_csv("runs/metrics/model_comparison.csv", index=False)


if __name__ == "__main__":
    main()