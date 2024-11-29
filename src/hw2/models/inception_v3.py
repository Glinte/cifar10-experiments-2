from typing import Callable

import torch
import torchvision
from torchvision.transforms import transforms


def get_inception_v3(num_classes: int = 1000) -> torchvision.models.Inception3:
    """
    Get an InceptionV3 model pretrained on ImageNet.

    If num_classes is not 1000, the classifier is replaced with a new one with num_classes output features.
    """
    model = torchvision.models.inception_v3(weights=torchvision.models.Inception_V3_Weights.IMAGENET1K_V1)
    if num_classes != 1000:  # Keep the original classifier and weights if num_classes is 1000
        model.fc = torch.nn.Linear(in_features=model.fc.in_features, out_features=num_classes)
    return model


def get_inception_v3_transform() -> Callable:
    """Get the transform for InceptionV3."""
    return transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
