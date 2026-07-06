import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


def find_best_threshold(scores, labels):

    thresholds = np.linspace(min(scores), max(scores), 200)

    best_f1 = 0
    best_t = thresholds[0]

    for t in thresholds:

        preds = (scores > t).astype(int)

        f1 = f1_score(labels, preds, zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    return best_t, best_f1


def main():

    df = pd.read_csv("runs/metrics/patchcore_scores.csv")

    scores = df["score"].values
    labels = df["label"].values

    # normalize scores (IMPORTANT)
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

    # ROC-AUC
    try:
        auc = roc_auc_score(labels, scores)
    except:
        auc = 0.0

    # best threshold
    threshold, best_f1 = find_best_threshold(scores, labels)

    preds = (scores > threshold).astype(int)

    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)

    print("\n=== PATCHCORE EVALUATION ===\n")

    print(f"ROC-AUC: {auc:.4f}")
    print(f"Best threshold: {threshold:.4f}")

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1: {f1:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(labels, preds))

    # save results
    with open("runs/metrics/patchcore_eval.json", "w") as f:
        import json
        json.dump({
            "roc_auc": float(auc),
            "threshold": float(threshold),
            "f1": float(f1)
        }, f, indent=4)


if __name__ == "__main__":
    main()