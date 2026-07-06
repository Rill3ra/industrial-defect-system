# src/models/patchcore_extractor.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class PatchCoreExtractor(nn.Module):
    """
    Feature extractor для PatchCore.
    Pretrained ResNet50, признаки из layer2 + layer3.
    layer2: [B, 512, 28, 28]
    layer3: [B, 1024, 14, 14] → upsample → [B, 1024, 28, 28]
    concat → [B, 1536, 28, 28]
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet50(weights=weights)

        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        feat2 = self.layer2(x)   # [B, 512,  28, 28]
        feat3 = self.layer3(feat2)  # [B, 1024, 14, 14]

        feat3_up = F.interpolate(
            feat3,
            size=feat2.shape[2:],
            mode="bilinear",
            align_corners=False
        )

        return torch.cat([feat2, feat3_up], dim=1)  # [B, 1536, 28, 28]
