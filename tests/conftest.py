import torch
from pytest_cases import fixture, parametrize
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2

from hw2 import PROJECT_ROOT
from hw2.util import CIFAR10_NORMALIZATION


@fixture(scope="session")
@parametrize("name", ["cifar10"])
def dataset(name) -> Dataset[tuple[torch.Tensor, int]]:
    """A fixture that returns a Dataset."""

    if name != "cifar10":
        raise ValueError(f"Unknown dataset name: {name}")

    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(*CIFAR10_NORMALIZATION),
    ])
    cifar10_train = CIFAR10(root=PROJECT_ROOT / "data", train=True, download=True, transform=transform)
    return cifar10_train
