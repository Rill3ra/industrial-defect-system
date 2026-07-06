import os
import json
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc

from src.data.dataset import DefectDataset
from src.data.transforms import get_val_transforms
from torch.utils.data import DataLoader

import torch

from src.models.model_factory import get_model


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

DATA_PATH = "data/splits/test.json"
METRICS_DIR = "runs/metrics"
PLOTS_DIR = "runs/plots"

os.makedirs(PLOTS_DIR, exist_ok=True)


dataset = DefectDataset(
    DATA_PATH,
    transform=get_val_transforms()
)

loader = DataLoader(dataset, batch_size=16, shuffle=False)


def get_probs(model):
    model.eval()
    model.to(device)

    probs_all = []
    labels_all = []

    with torch.no_grad():
        for batch in loader:

            x = batch["image"].to(device)
            y = batch["label"].cpu().numpy()

            logits = model(x)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

            probs_all.extend(probs)
            labels_all.extend(y)

    return np.array(labels_all), np.array(probs_all)


models = {
    "resnet50": get_model("resnet50"),
    "densenet121": get_model("densenet121"),
    "efficientnet_b0": get_model("efficientnet_b0"),
    "vit_b16": get_model("vit_b16"),
}


plt.figure(figsize=(8, 6))


for name, model in models.items():

    checkpoint = torch.load(
        f"runs/checkpoints/{name}.pth",
        map_location=device
    )

    model.load_state_dict(checkpoint)

    y_true, y_score = get_probs(model)

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC = {roc_auc:.3f})"
    )


plt.plot([0, 1], [0, 1], "k--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves Comparison")
plt.legend()

output_path = os.path.join(PLOTS_DIR, "roc_curves.png")

plt.savefig(output_path, dpi=300)

print("Saved:", output_path)