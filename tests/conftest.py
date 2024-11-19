import torch
from pytest_cases import fixture, parametrize
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2

from hw2 import PROJECT_ROOT
from hw2.util import CIFAR10_NORMALIZATION


@fixture(scope="session")
@parametrize("dataset", ["cifar10"])
def dataloader(dataset: str) -> DataLoader:
    """A fixture that returns a DataLoader."""

    if dataset != "cifar10":
        raise ValueError(f"Unknown dataset: {dataset}")

    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(*CIFAR10_NORMALIZATION),
    ])
    cifar10_train = CIFAR10(root=PROJECT_ROOT / "data", train=True, download=True, transform=transform)
    return DataLoader(cifar10_train, batch_size=65536, shuffle=False, num_workers=2)
