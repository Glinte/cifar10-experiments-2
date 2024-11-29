"""A simple multi-layer perceptron."""
from __future__ import annotations

import logging
from functools import partial
from typing import TYPE_CHECKING, Sequence

import torch
import wandb
from sklearn import metrics
from torch import nn
from torch.nn import CrossEntropyLoss

from hw2 import PROJECT_ROOT, CIFAR10_NORMALIZATION
from hw2.open_set import one_minus_max_of_prob
from hw2.utils import train_on_cifar, validate_on_cifar, validate_on_open_set

if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    from torch import optim


logger = logging.getLogger(__name__)


class MLP(nn.Module):
    """A simple multi-layer perceptron."""

    def __init__(self, input_dim: int, hidden_dims: list[int], output_dim: int) -> None:
        """
        Initialize the MLP.

        Args:
            input_dim: The number of input features.
            hidden_dims: The number of hidden units in each layer.
            output_dim: The number of output classes.
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims

        self.layers = nn.ModuleList()
        prev_dim = input_dim
        for dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, dim))
            prev_dim = dim
        self.layers.append(nn.Linear(prev_dim, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""

        x = x.view(x.size(0), -1)  # flatten the input
        for layer in self.layers[:-1]:
            x = nn.functional.relu(layer(x))
        x = self.layers[-1](x)
        return x

def main():
    from torchvision.transforms import v2

    logging.basicConfig(level=logging.INFO)

    epochs = 150
    dims = [3 * 32 * 32, 1024, 512, 256, 128, 64, 10]

    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(*CIFAR10_NORMALIZATION),
    ])

    run = wandb.init(
        project="CISC3027 hw2",
        job_type="test",
        config={
            "model": "MLP",
        }
    )

    model = MLP(input_dim=dims[0], hidden_dims=dims[1:-1], output_dim=dims[-1])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    train_on_cifar(model, optimizer, criterion, None, transform, epochs=epochs, device=torch.device("cuda"), log_run=True)
    torch.save(model.state_dict(), PROJECT_ROOT / f"models/mlp_{"_".join(map(str, dims))}_epoch{epochs}_2.pth")

if __name__ == "__main__":
    # main()
    from torchvision.transforms import v2
    model = MLP(input_dim=3072, hidden_dims=[1024, 512, 256, 128, 64], output_dim=10)
    model.load_state_dict(torch.load(PROJECT_ROOT / "models/mlp_3072_1024_512_256_128_64_10_epoch150_2.pth"))
    transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((32, 32)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(*CIFAR10_NORMALIZATION),
    ])
    results = validate_on_cifar(model, CrossEntropyLoss(), transform, device=torch.device("cuda"), additional_metrics=[partial(metrics.classification_report, digits=4)])
    for metric, value in results.items():
        print(f"{metric}: {value}")
    results = validate_on_open_set(model, open_set_prob_fn=one_minus_max_of_prob, transform=transform, device=torch.device("cuda"))
    print(results)
