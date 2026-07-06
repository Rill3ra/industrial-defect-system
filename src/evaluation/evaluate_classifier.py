# src/evaluation/evaluate_classifier.py

import argparse
import json
import os

import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from src.data.dataset import DefectDataset
from src.data.transforms import get_val_transforms
from src.models.model_factory import get_model


parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    type=str,
    required=True
)

args = parser.parse_args()

MODEL_NAME = args.model


device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print("Device:", device)
print("Model:", MODEL_NAME)


dataset = DefectDataset(
    "data/splits/test.json",
    transform=get_val_transforms()
)

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=False
)


model = get_model(MODEL_NAME)

model.load_state_dict(
    torch.load(
        f"runs/checkpoints/{MODEL_NAME}.pth",
        map_location=device
    )
)

model.to(device)
model.eval()


preds_all = []
labels_all = []
probs_all = []


with torch.no_grad():

    for batch in loader:

        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        outputs = model(images)

        probs = F.softmax(
            outputs,
            dim=1
        )[:, 1]

        preds = outputs.argmax(1)

        preds_all.extend(
            preds.cpu().numpy()
        )

        labels_all.extend(
            labels.cpu().numpy()
        )

        probs_all.extend(
            probs.cpu().numpy()
        )


accuracy = accuracy_score(
    labels_all,
    preds_all
)

precision = precision_score(
    labels_all,
    preds_all,
    zero_division=0
)

recall = recall_score(
    labels_all,
    preds_all,
    zero_division=0
)

f1 = f1_score(
    labels_all,
    preds_all,
    zero_division=0
)

roc_auc = roc_auc_score(
    labels_all,
    probs_all
)


print("\n=== EVALUATION ===\n")

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1:", f1)
print("ROC-AUC:", roc_auc)

print("\nClassification Report\n")

print(
    classification_report(
        labels_all,
        preds_all,
        zero_division=0
    )
)

print("\nConfusion Matrix\n")

print(
    confusion_matrix(
        labels_all,
        preds_all
    )
)


os.makedirs(
    "runs/metrics",
    exist_ok=True
)

with open(
    f"runs/metrics/{MODEL_NAME}.json",
    "w"
) as f:

    json.dump(
        {
            "model": MODEL_NAME,
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "roc_auc": float(roc_auc)
        },
        f,
        indent=4
    )

print(
    f"\nMetrics saved: runs/metrics/{MODEL_NAME}.json"
)