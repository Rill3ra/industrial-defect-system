import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from src.inference.pipeline import predict
from src.data.transforms import get_val_transforms

import matplotlib.patches as patches


PLOT_DIR = "runs/visualizations/pipeline"
os.makedirs(PLOT_DIR, exist_ok=True)


transform = get_val_transforms()


# ----------------------------
# HEATMAP OVERLAY
# ----------------------------
def overlay_heatmap(image, heatmap):

    image = np.array(image).astype(np.float32) / 255.0
    heatmap = heatmap.cpu().numpy()

    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.imshow(heatmap, cmap="jet", alpha=0.5)
    plt.axis("off")


# ----------------------------
# SAVE RESULT
# ----------------------------
def save_result(image_path, result):

    image = Image.open(image_path).convert("RGB")

    heatmap = result["heatmap"]

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.imshow(image)
    ax.imshow(heatmap.cpu().numpy(), cmap="jet", alpha=0.45)

    bbox = result.get("bbox")

    if bbox is not None:
        x1, y1, x2, y2 = bbox

        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            edgecolor="red",
            facecolor="none"
        )

        ax.add_patch(rect)

    ax.set_title(
        f"{result['decision']} | score={result['patchcore_score']:.3f}"
    )

    ax.axis("off")

    name = os.path.basename(image_path).replace(".png", "_vis.png")
    save_path = os.path.join(PLOT_DIR, name)

    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()

    return save_path


# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":

    test_images = [
        "data/raw/mvtec_ad/screw/test/good/000.png",
        "data/raw/mvtec_ad/screw/test/manipulated_front/000.png"
    ]

    for path in test_images:

        if not os.path.exists(path):
            print("[SKIP]", path)
            continue

        result = predict(path)

        save_path = save_result(path, result)

        print("Saved:", save_path)