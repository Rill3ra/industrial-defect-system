# src/training/train_classifier.py
import argparse
import json
import os
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.data.dataset import DefectDataset
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.model_factory import get_model

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--epochs", type=int, default=30)
args = parser.parse_args()

MODEL_NAME = args.model
EPOCHS = args.epochs

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Проверь строку device в train_classifier.py — должно быть:
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
print("Model:", MODEL_NAME)
print("Epochs:", EPOCHS)

# ─────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────
train_dataset = DefectDataset("data/splits/train.json", transform=get_train_transforms())
val_dataset   = DefectDataset("data/splits/val.json",   transform=get_val_transforms())

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=8, shuffle=False, num_workers=0)

# ─────────────────────────────────────────
# CLASS WEIGHTS
# ─────────────────────────────────────────
label_list = [s["label"] for s in train_dataset.samples]
counter = Counter(label_list)
num_normal = counter[0]
num_defect = counter[1]

weights = torch.tensor([1.0, num_normal / num_defect], dtype=torch.float32).to(device)
print(f"\nClass distribution: {counter}")
print(f"Class weights: {weights}")

# ─────────────────────────────────────────
# MODEL — pretrained
# ─────────────────────────────────────────
model = get_model(MODEL_NAME, pretrained=True)
model = model.to(device)

criterion = nn.CrossEntropyLoss(weight=weights)

optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

# cosine annealing — плавно снижает lr до нуля
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

# ─────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────
best_f1 = 0.0
history = []

os.makedirs("runs/checkpoints", exist_ok=True)
os.makedirs("runs/logs", exist_ok=True)

for epoch in range(EPOCHS):

    model.train()
    train_preds, train_labels_list = [], []
    running_loss = 0.0

    for batch in train_loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = outputs.argmax(1)
        train_preds.extend(preds.cpu().numpy())
        train_labels_list.extend(labels.cpu().numpy())

    scheduler.step()

    train_acc = accuracy_score(train_labels_list, train_preds)
    train_f1  = f1_score(train_labels_list, train_preds, zero_division=0)

    # ── VALIDATION ──
    model.eval()
    val_preds, val_labels_list = [], []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = model(images)
            preds = outputs.argmax(1)
            val_preds.extend(preds.cpu().numpy())
            val_labels_list.extend(labels.cpu().numpy())

    val_acc  = accuracy_score(val_labels_list, val_preds)
    val_prec = precision_score(val_labels_list, val_preds, zero_division=0)
    val_rec  = recall_score(val_labels_list, val_preds, zero_division=0)
    val_f1   = f1_score(val_labels_list, val_preds, zero_division=0)
    current_lr = scheduler.get_last_lr()[0]

    print(f"\nEpoch {epoch+1}/{EPOCHS} | LR: {current_lr:.6f}")
    print(f"  Loss: {running_loss:.4f}")
    print(f"  Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f}")
    print(f"  Val Acc: {val_acc:.4f} | P: {val_prec:.4f} | R: {val_rec:.4f} | F1: {val_f1:.4f}")

    history.append({
        "epoch": epoch + 1,
        "lr": current_lr,
        "train_acc": float(train_acc),
        "train_f1": float(train_f1),
        "val_acc": float(val_acc),
        "val_precision": float(val_prec),
        "val_recall": float(val_rec),
        "val_f1": float(val_f1),
    })

    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(model.state_dict(), f"runs/checkpoints/{MODEL_NAME}.pth")
        print("  Saved best model")

# ─────────────────────────────────────────
# SAVE LOGS
# ─────────────────────────────────────────
with open(f"runs/logs/{MODEL_NAME}.json", "w") as f:
    json.dump({"model": MODEL_NAME, "best_f1": float(best_f1), "history": history}, f, indent=4)

print(f"\nTraining finished. Best F1: {best_f1:.4f}")