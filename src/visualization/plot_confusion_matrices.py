import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

from torch.utils.data import DataLoader

from src.data.dataset import DefectDataset
from src.data.transforms import get_val_transforms
from src.models.model_factory import get_model


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

DATA_PATH = "data/splits/test.json"

PLOTS_DIR = "runs/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


dataset = DefectDataset(
    DATA_PATH,
    transform=get_val_transforms()
)

loader = DataLoader(dataset, batch_size=16, shuffle=False)


def evaluate_model(model):
    model.eval()
    model.to(device)

    preds_all = []
    labels_all = []

    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device)
            y = batch["label"].cpu().numpy()

            logits = model(x)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            preds_all.extend(preds)
            labels_all.extend(y)

    return np.array(labels_all), np.array(preds_all)


def plot_cm(cm, title, path):
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, cmap="Blues")

    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.colorbar()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


models = {
    "resnet50": get_model("resnet50"),
    "densenet121": get_model("densenet121"),
    "efficientnet_b0": get_model("efficientnet_b0"),
    "vit_b16": get_model("vit_b16"),
}


for name, model in models.items():

    print(f"Processing {name}...")

    model.load_state_dict(
        torch.load(
            f"runs/checkpoints/{name}.pth",
            map_location=device
        )
    )

    y_true, y_pred = evaluate_model(model)

    cm = confusion_matrix(y_true, y_pred)

    save_path = os.path.join(PLOTS_DIR, f"{name}_cm.png")

    plot_cm(cm, name, save_path)

    print("Saved:", save_path)