"""Tests of the visualization module."""
import pytest
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2

from hw2 import PROJECT_ROOT
from hw2.util import CIFAR10_NORMALIZATION
from hw2.visualization import generate_distinct_colors, PCA_visualization


@pytest.mark.parametrize("n_colors", [1, 5, 10, 20, 50])
def test_generate_distinct_colors(n_colors):
    colors = generate_distinct_colors(n_colors)
    assert len(colors) == n_colors
    for color in colors:
        assert len(color) == 4  # RGBA
        assert all(0 <= c <= 1 for c in color)


def test_PCA_visualization():
    """Test the PCA_visualization function."""
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(*CIFAR10_NORMALIZATION),
    ])
    cifar10_train = CIFAR10(root=PROJECT_ROOT / "data", train=True, download=True, transform=transform)
    cifar10_train_loader = DataLoader(cifar10_train, batch_size=65536, shuffle=False, num_workers=2)
    X, y = next(iter(cifar10_train_loader))
    X = X.reshape(X.shape[0], -1)
    PCA_visualization(X, y, cifar10_train.classes)
