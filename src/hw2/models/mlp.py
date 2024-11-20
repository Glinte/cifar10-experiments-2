"""A simple multi-layer perceptron."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Sequence

import torch
import wandb
from sklearn import metrics
from torch import nn

from hw2 import PROJECT_ROOT
from hw2.util import CIFAR10_NORMALIZATION

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

    def train_loop(
        self,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        epochs: int = 2,
        device: torch.device = torch.device("cuda"),
        save_to: str | None = None,
    ) -> None:
        """Train the model."""

        run = wandb.init(
            project="CISC3027 hw2",
            job_type="train",
            config={
                "model": "MLP",
                "epochs": epochs,
                "optimizer": optimizer.__class__.__name__,
                "learning_rate": optimizer.defaults.get("lr"),
                "hidden_dims": self.hidden_dims,
            }
        )

        self.to(device=device)
        try:
            for epoch in range(epochs):
                running_loss = 0.0
                for i, data in enumerate[tuple[torch.Tensor, torch.Tensor]](dataloader):
                    # get the inputs; data is a list of [inputs, labels]
                    inputs, labels = data

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward + backward + optimize
                    outputs = self(inputs.to(device=device))
                    loss = criterion(outputs, labels.to(device=device))
                    loss.backward()
                    optimizer.step()

                    # print statistics
                    running_loss += loss.item()
                    if i % 200 == 199:  # print every 500 mini-batches
                        accuracy = metrics.accuracy_score(labels.cpu().numpy(), outputs.argmax(dim=1).cpu().numpy())
                        run.log({"accuracy": accuracy, "batch_loss": loss, "epoch": epoch, "batch": i})
                        print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}")
                        running_loss = 0.0
        except KeyboardInterrupt:
            logger.info("Interrupted training.")
        finally:
            if save_to is not None:
                torch.save(self.state_dict(), save_to)
                logger.info(f"Saved model to {save_to}")

        run.finish()

    def test_loop(self, dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]], device: torch.device = torch.device("cuda"), target_names: Sequence[str] | None = None) -> float:
        """Test the model."""

        run = wandb.init(
            project="CISC3027 hw2",
            job_type="test",
            config={
                "model": "MLP",
            }
        )

        self.eval()
        predictions = []
        true_labels = []
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self(inputs)
                _, predicted = torch.max(outputs, 1)
                predictions.extend(predicted.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        accuracy = metrics.accuracy_score(true_labels, predictions)
        print(metrics.classification_report(true_labels, predictions, target_names=target_names, digits=4))
        run.log({"accuracy": accuracy})
        run.finish()
        return accuracy


def main():
    from torch.utils.data import DataLoader
    from torchvision.datasets import CIFAR10
    from torchvision.transforms import v2

    epochs = 150
    dims = [3 * 32 * 32, 3072, 3072, 3072, 3072, 3072, 10]

    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(*CIFAR10_NORMALIZATION),
    ])
    cifar10_train_dataset = CIFAR10(root=PROJECT_ROOT / "data", train=True, download=True, transform=transform)
    cifar10_test_dataset = CIFAR10(root=PROJECT_ROOT / "data", train=False, download=True, transform=transform)

    model = MLP(input_dim=dims[0], hidden_dims=dims[1:-1], output_dim=dims[-1])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    train_loader = DataLoader(cifar10_train_dataset, batch_size=32, shuffle=True, num_workers=2)

    model.train_loop(criterion, optimizer, train_loader, epochs=epochs)
    model.test_loop(DataLoader(cifar10_test_dataset, batch_size=65536, shuffle=False, num_workers=2), target_names=cifar10_test_dataset.classes)
    torch.save(model.state_dict(), PROJECT_ROOT / f"models/mlp_{"_".join(map(str, dims))}_epoch{epochs}.pth")

if __name__ == "__main__":
    main()
