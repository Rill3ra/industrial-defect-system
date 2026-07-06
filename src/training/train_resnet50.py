import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import os

import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from torchvision import models

from src.data.dataset import DefectDataset
from src.data.transforms import get_train_transforms, get_val_transforms


# =========================
# DEVICE
# =========================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Device:", device)


# =========================
# DATA
# =========================
train_dataset = DefectDataset(
    json_path="data/splits/train.json",
    transform=get_train_transforms()
)

val_dataset = DefectDataset(
    json_path="data/splits/val.json",
    transform=get_val_transforms()
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)


# =========================
# CLASS WEIGHTS (IMPORTANT)
# =========================
labels = [item["label"] for item in train_dataset.samples]
class_counts = np.bincount(labels)

print("Class counts:", class_counts)

weights = class_counts.sum() / (2 * class_counts)
weights = torch.tensor(weights, dtype=torch.float).to(device)

print("Class weights:", weights)


# =========================
# MODEL
# =========================
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)


# =========================
# LOSS + OPTIMIZER
# =========================
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)


# =========================
# TRAIN
# =========================
def train_one_epoch():
    model.train()

    all_preds = []
    all_labels = []

    total_loss = 0

    for batch in train_loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    return total_loss / len(train_loader), acc, f1


# =========================
# EVAL
# =========================
def evaluate():
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    return acc, f1, all_labels, all_preds


# =========================
# TRAIN LOOP
# =========================
EPOCHS = 10
best_f1 = 0

os.makedirs("runs/checkpoints", exist_ok=True)
os.makedirs("runs/visualizations/confusion_matrices", exist_ok=True)

for epoch in range(EPOCHS):

    train_loss, train_acc, train_f1 = train_one_epoch()
    val_acc, val_f1, y_true, y_pred = evaluate()

    scheduler.step()

    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Train Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
    print(f"Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

    # save best model
    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(model.state_dict(), "runs/checkpoints/resnet50.pth")
        print("Saved best model")


# =========================
# FINAL REPORT
# =========================
print("\nClassification report:")
print(classification_report(y_true, y_pred))


# =========================
# CONFUSION MATRIX
# =========================
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("ResNet50 Confusion Matrix")

plt.savefig("runs/visualizations/confusion_matrices/resnet50.png")
plt.close()

print("\nConfusion matrix saved.")