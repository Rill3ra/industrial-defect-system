# src/training/train_patchcore.py
import torch
import numpy as np
from torch.utils.data import DataLoader
from src.data.dataset import DefectDataset
from src.data.transforms import get_val_transforms
from src.models.patchcore_extractor import PatchCoreExtractor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ─────────────────────────────────────────
# DATASET — только нормальные для memory bank
# ─────────────────────────────────────────
dataset = DefectDataset(
    "data/splits/train.json",
    transform=get_val_transforms(),
    only_normal=True
)
loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
print(f"Normal train samples: {len(dataset)}")

# ─────────────────────────────────────────
# MODEL — pretrained ResNet50
# ─────────────────────────────────────────
model = PatchCoreExtractor(pretrained=True).to(device)
model.eval()

# ─────────────────────────────────────────
# BUILD MEMORY BANK
# ─────────────────────────────────────────
memory_bank = []

with torch.no_grad():
    for i, batch in enumerate(loader):
        x = batch["image"].to(device)
        feat_map = model(x)          # [B, 1536, 28, 28]
        B, C, H, W = feat_map.shape

        # патчи: каждый пиксель feature map = один патч
        patches = feat_map.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, 1536]
        memory_bank.append(patches.cpu())

        if i % 10 == 0:
            print(f"  batch {i}/{len(loader)}")

memory_bank = torch.cat(memory_bank, dim=0)
print(f"\nRaw memory bank: {memory_bank.shape}")

# ─────────────────────────────────────────
# SUBSAMPLING — случайная выборка патчей
# Уменьшаем до 50k для скорости инференса
# ─────────────────────────────────────────
MAX_PATCHES = 50_000

if memory_bank.shape[0] > MAX_PATCHES:
    idx = torch.randperm(memory_bank.shape[0])[:MAX_PATCHES]
    memory_bank = memory_bank[idx]
    print(f"Subsampled memory bank: {memory_bank.shape}")

# ─────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────
import os
os.makedirs("runs/checkpoints", exist_ok=True)
os.makedirs("runs/metrics", exist_ok=True)

torch.save(memory_bank, "runs/checkpoints/patchcore_memory_bank.pt")
print("Saved memory bank →", memory_bank.shape)

# ─────────────────────────────────────────
# COMPUTE THRESHOLD на train-нормальных
# ─────────────────────────────────────────
print("\nComputing threshold on normal samples...")

scores = []
with torch.no_grad():
    for batch in DataLoader(dataset, batch_size=4, shuffle=False):
        x = batch["image"].to(device)
        feat_map = model(x)
        B, C, H, W = feat_map.shape
        patches = feat_map.permute(0, 2, 3, 1).reshape(-1, C).cpu()

        dist = torch.cdist(patches, memory_bank)
        score = dist.min(dim=1)[0].max().item()  # max patch distance = image score
        scores.append(score)

scores = torch.tensor(scores)
stats = {
    "mean":         scores.mean().item(),
    "std":          scores.std().item(),
    "threshold_90": torch.quantile(scores, 0.90).item(),
    "threshold_95": torch.quantile(scores, 0.95).item(),
    "threshold_99": torch.quantile(scores, 0.99).item(),
}

torch.save(stats, "runs/metrics/patchcore_stats.pt")
print("\nPATCHCORE STATS:")
for k, v in stats.items():
    print(f"  {k}: {v:.4f}")
print("\nDone!")