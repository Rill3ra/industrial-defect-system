import numpy as np
import torch


def heatmap_to_bbox(heatmap: torch.Tensor, threshold: float = 0.5):
    """
    Converts PatchCore heatmap -> bounding box
    returns: (x1, y1, x2, y2)
    """

    hmap = heatmap.detach().cpu().numpy()

    mask = hmap > threshold

    if mask.sum() == 0:
        return None

    coords = np.argwhere(mask)

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    return int(x_min), int(y_min), int(x_max), int(y_max)