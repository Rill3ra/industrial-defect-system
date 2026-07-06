import json
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from src.models.model_factory import get_model
from src.models.patchcore_extractor import PatchCoreExtractor
from src.data.transforms import get_val_transforms


# ----------------------------
# DEVICE
# ----------------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ----------------------------
# CLASSIFIER
# ----------------------------
MODEL_NAME = "densenet121"

classifier = get_model(MODEL_NAME)
classifier.load_state_dict(
    torch.load(f"runs/checkpoints/{MODEL_NAME}.pth", map_location=device)
)
classifier.to(device)
classifier.eval()


# ----------------------------
# PATCHCORE
# ----------------------------
patchcore_bank = torch.load(
    "runs/checkpoints/patchcore_memory_bank.pt",
    map_location=device
)

extractor = PatchCoreExtractor().to(device)
extractor.eval()


# ----------------------------
# PATCHCORE STATS (FIXED)
# ----------------------------
stats = torch.load("runs/metrics/patchcore_stats.pt")

patchcore_threshold = stats["threshold_95"]
patchcore_mean = stats["mean"]
patchcore_std = stats["std"]


transform = get_val_transforms()


# ----------------------------
# FEATURE EXTRACT
# ----------------------------
def extract_feature_map(x):
    with torch.no_grad():
        return extractor(x)


# ----------------------------
# PATCHCORE SCORE (RAW)
# ----------------------------
def patchcore_score(feat_map, bank):
    B, C, H, W = feat_map.shape

    patches = feat_map.permute(0, 2, 3, 1).reshape(-1, C)

    dist = torch.cdist(patches, bank)

    return dist.min().item()


# ----------------------------
# PATCHCORE HEATMAP
# ----------------------------
def patchcore_heatmap(feat_map, bank):

    B, C, H, W = feat_map.shape

    patches = feat_map.permute(0, 2, 3, 1).reshape(-1, C)

    dist = torch.cdist(patches, bank)

    min_dist = dist.min(dim=1)[0]

    heatmap = min_dist.reshape(H, W)

    # normalize only for visualization
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    return heatmap


# ----------------------------
# PREDICT
# ----------------------------
def predict(image_path):

    image = Image.open(image_path).convert("RGB")
    x = transform(image).unsqueeze(0).to(device)

    # ---------------- classifier ----------------
    with torch.no_grad():
        logits = classifier(x)
        probs = F.softmax(logits, dim=1)[0]
        class_prob = probs[1].item()

    # ---------------- patchcore ----------------
    with torch.no_grad():
        feat_map = extract_feature_map(x)

        pscore = patchcore_score(feat_map, patchcore_bank)

        heatmap = patchcore_heatmap(feat_map, patchcore_bank)

    # ---------------- decision ----------------
    label = int(pscore > patchcore_threshold)
    decision = "defect" if label == 1 else "normal"

    return {
        "final_label": label,
        "decision": decision,
        "classifier_prob": class_prob,
        "patchcore_score": pscore,
        "heatmap": heatmap
    }


# ----------------------------
# VISUALIZATION
# ----------------------------
def show_heatmap(image_path, heatmap):

    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))

    image = np.array(image)
    heatmap = heatmap.cpu().numpy()

    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.imshow(heatmap, cmap="jet", alpha=0.5)
    plt.axis("off")
    plt.colorbar()
    plt.show()


# ----------------------------
# TEST
# ----------------------------
if __name__ == "__main__":

    test_image = "data/raw/mvtec_ad/screw/test/good/000.png"

    result = predict(test_image)

    print("\n=== PREDICTION ===")
    print({
        "final_label": result["final_label"],
        "decision": result["decision"],
        "classifier_prob": result["classifier_prob"],
        "patchcore_score": result["patchcore_score"]
    })

    show_heatmap(test_image, result["heatmap"])