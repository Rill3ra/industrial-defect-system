#src/inference/pipeline.py
import torch
import numpy as np
from PIL import Image

from src.models.patchcore_extractor import PatchCoreExtractor
from src.data.transforms import get_val_transforms
from src.utils.heatmap import patchcore_heatmap
from src.models.patchcore_postprocess import heatmap_to_bbox
from src.models.model_factory import get_model


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

transform = get_val_transforms()

# ---------------- models ----------------
classifier = get_model("densenet121").to(device)
classifier.load_state_dict(torch.load(
    "runs/checkpoints/densenet121.pth",
    map_location=device
))
classifier.eval()

extractor = PatchCoreExtractor().to(device)
extractor.eval()

patchcore_bank = torch.load(
    "runs/checkpoints/patchcore_memory_bank.pt",
    map_location=device
)

patchcore_threshold = 0.08


# ---------------- core functions ----------------
def extract_feature_map(x):
    with torch.no_grad():
        return extractor(x)


def patchcore_score(feat_map, bank):
    B, C, H, W = feat_map.shape
    patches = feat_map.permute(0, 2, 3, 1).reshape(-1, C)
    dist = torch.cdist(patches, bank)
    return dist.min().item()


# ---------------- main ----------------
def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    x = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Classifier
        logits = classifier(x)
        probs = torch.softmax(logits, dim=1)[0]
        classifier_prob = probs[1].item()
        
        # PatchCore
        feat_map = extract_feature_map(x)
        score = patchcore_score(feat_map, patchcore_bank)
        
        # Heatmap
        heatmap = patchcore_heatmap(feat_map, patchcore_bank)
        
        # BBox
        bbox = heatmap_to_bbox(heatmap, threshold_percentile=90)  # было 95 → делаем чуть мягче

    # Final decision
    if score > patchcore_threshold:
        final_label = 1
        decision = "defect"
    else:
        final_label = int(classifier_prob > 0.5)
        decision = "defect" if final_label == 1 else "normal"

    return {
        "final_label": final_label,
        "decision": decision,
        "classifier_prob": float(classifier_prob),
        "patchcore_score": float(score),
        "heatmap": heatmap.tolist(),          # ← важно: .tolist()
        "bbox": bbox
    }
