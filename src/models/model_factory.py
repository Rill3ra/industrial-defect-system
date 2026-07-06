# src/models/model_factory.py
from torchvision import models
import torch.nn as nn


def get_model(model_name: str, pretrained: bool = True):
    """
    Возвращает модель с pretrained весами ImageNet.
    Последний слой заменяется под 2 класса (normal / defect).
    """

    if pretrained:
        print(f"  Loading pretrained weights for {model_name}")

    if model_name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, 2)

    elif model_name == "densenet121":
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, 2)

    elif model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)

    elif model_name == "vit_b16":
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.vit_b_16(weights=weights)
        model.heads.head = nn.Linear(model.heads.head.in_features, 2)

    elif model_name == "mobilenet_v3":
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model
