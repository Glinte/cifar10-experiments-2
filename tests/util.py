import numpy as np
import torch
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2

from hw2 import PROJECT_ROOT
from hw2.util import find_mean_and_stddev, CIFAR10_NORMALIZATION


def test_cifar_10_normalization() -> None:
    """Test that the CIFAR-10 normalization is correct."""
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])
    cifar10_normalization = find_mean_and_stddev(CIFAR10(root=PROJECT_ROOT / "data", train=True, transform=transform))
    assert np.allclose(cifar10_normalization, CIFAR10_NORMALIZATION, atol=1e-3)
