#src/models/patchcore_postprocess.py
import torch
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from skimage.measure import label, regionprops


def heatmap_to_bbox(heatmap: torch.Tensor, threshold_percentile: float = 95):
    """
    heatmap: [H, W] tensor (0–1 normalized)
    returns: bbox (x1, y1, x2, y2) or None
    """

    hm = heatmap.detach().cpu().numpy()

    # 1. smoothing (IMPORTANT)
    hm = gaussian_filter(hm, sigma=2)

    # 2. threshold (adaptive, NOT fixed)
    thr = np.percentile(hm, threshold_percentile)
    binary = hm > thr

    # 3. connected components
    labeled = label(binary)

    if labeled.max() == 0:
        return None

    # 4. pick largest component
    regions = regionprops(labeled)
    largest = max(regions, key=lambda r: r.area)

    minr, minc, maxr, maxc = largest.bbox

    return {
        "x1": int(minc),
        "y1": int(minr),
        "x2": int(maxc),
        "y2": int(maxr),
        "score": float(hm.max())
    }