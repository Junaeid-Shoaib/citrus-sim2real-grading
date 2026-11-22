"""ResNet18 transfer-learning model and the device helper."""

import torch
from torch import nn
from torchvision import models

NUM_CLASSES = 3


def get_device():
    # Apple Silicon GPU when available; CPU keeps the code runnable anywhere else
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def build_model():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # Replace the 1000-class ImageNet head with a fresh 3-grade head
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


def set_backbone_trainable(model, trainable):
    for name, param in model.named_parameters():
        if not name.startswith("fc."):
            param.requires_grad = trainable
