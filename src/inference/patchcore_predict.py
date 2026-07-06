import os
import csv
import torch
from torch.utils.data import DataLoader

from src.data.dataset import DefectDataset
from src.data.transforms import get_val_transforms
from src.models.patchcore import PatchCore

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Device:", device)

test_dataset = DefectDataset(
    json_path="data/splits/test.json",
    transform=get_val_transforms()
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False
)

model = PatchCore(device)

model.memory_bank = torch.load(
    "runs/checkpoints/patchcore_memory_bank.pt"
)

os.makedirs("runs/metrics", exist_ok=True)

csv_path = "runs/metrics/patchcore_scores.csv"

with open(csv_path, "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
        "label",
        "score",
        "category",
        "defect_type"
    ])

    model.eval()

    with torch.no_grad():

        for batch in test_loader:

            image = batch["image"].to(device)

            score = model.compute_anomaly_score(image)

            writer.writerow([
                int(batch["label"].item()),
                float(score.item()),
                batch["category"][0],
                batch["defect_type"][0]
            ])

print("\nSaved:")
print(csv_path)