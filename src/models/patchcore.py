# src/models/patchcore.py
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import numpy as np


class PatchCore(nn.Module):
    def __init__(self, device):
        super().__init__()

        self.device = device

        # backbone (feature extractor)
        self.backbone = models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

        self.backbone.to(device)
        self.backbone.eval()

        self.memory_bank = None

    def extract_features(self, x):
        with torch.no_grad():
            feats = self.backbone(x)
            feats = feats.mean(dim=[2, 3])  # global pooling
        return feats

    def build_memory_bank(self, dataloader):
        features = []

        for batch in dataloader:
            x = batch["image"].to(self.device)
            f = self.extract_features(x)
            features.append(f.cpu())

        self.memory_bank = torch.cat(features, dim=0)

        print("Memory bank size:", self.memory_bank.shape)

    def compute_anomaly_score(self, x):
        f = self.extract_features(x)

        dists = torch.cdist(f.cpu(), self.memory_bank)
        score = torch.min(dists, dim=1)[0]

        return score