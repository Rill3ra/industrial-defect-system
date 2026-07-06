import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from src.data.transforms import get_val_transforms
from src.models.patchcore_extractor import PatchCoreExtractor


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

PLOTS_DIR = "runs/plots/patchcore"
os.makedirs(PLOTS_DIR, exist_ok=True)


# ----------------------------
# MODEL + MEMORY
# ----------------------------
extractor = PatchCoreExtractor().to(device)
extractor.eval()

patchcore_bank = torch.load(
    "runs/checkpoints/patchcore_memory_bank.pt",
    map_location=device
)


# ----------------------------
# HEATMAP FUNCTION
# ----------------------------
def compute_heatmap(feat_map, bank):

    B, C, H, W = feat_map.shape

    patches = feat_map.permute(0, 2, 3, 1).reshape(-1, C)

    dist = torch.cdist(patches, bank)

    min_dist = dist.min(dim=1)[0]

    heatmap = min_dist.reshape(H, W)

    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    return heatmap.cpu().numpy()


# ----------------------------
# PLOT FUNCTION
# ----------------------------
def overlay(image, heatmap, save_path):

    plt.figure(figsize=(6, 6))

    plt.imshow(image)
    plt.imshow(heatmap, cmap="jet", alpha=0.5)

    plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# ----------------------------
# PROCESS SINGLE IMAGE
# ----------------------------
def process_image(image_path):

    transform = get_val_transforms()

    image = Image.open(image_path).convert("RGB")

    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        feat_map = extractor(x)

    heatmap = compute_heatmap(feat_map, patchcore_bank)

    return image, heatmap


# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":

    test_images = [
        "data/raw/mvtec_ad/screw/test/good/000.png",
        "data/raw/mvtec_ad/screw/test/manipulated_front/000.png"  # пример дефекта (реальный класс MVTec screw)
    ]

    for path in test_images:

        if not os.path.exists(path):
            print(f"[SKIP] File not found: {path}")
            continue

        image, heatmap = process_image(path)

        name = os.path.basename(path)

        save_path = os.path.join(PLOTS_DIR, name)

        overlay(image, heatmap, save_path)

        print("Saved:", save_path)