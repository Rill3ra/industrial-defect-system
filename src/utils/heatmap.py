# src/utils/heatmap.py

import torch


def patchcore_heatmap(feat_map, bank):

    B, C, H, W = feat_map.shape
    patches = feat_map.permute(0, 2, 3, 1).reshape(-1, C)

    dist = torch.cdist(patches, bank)
    min_dist = dist.min(dim=1)[0]

    heatmap = min_dist.reshape(H, W)

    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    return heatmap