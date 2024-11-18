from torchvision.datasets import CIFAR10

from hw2 import PROJECT_ROOT
from hw2.util import find_mean_and_stddev, CIFAR10_NORMALIZATION


def test_cifar_10_normalization() -> None:
    """Test that the CIFAR-10 normalization is correct."""
    cifar10_normalization = find_mean_and_stddev(CIFAR10(root=PROJECT_ROOT / "data", train=True, transform=None))
    assert cifar10_normalization == CIFAR10_NORMALIZATION
