"""Basic LeNet-5 CNN model for CIFAR-10 classification."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F


if TYPE_CHECKING:
    from torch.utils.data import DataLoader


logger = logging.getLogger(__name__)


class CNN(nn.Module):
    def __init__(self, channels: int) -> None:
        """Initialize the CNN.

        Args:
            channels: Number of channels in the input images.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels * 2, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(channels * 2, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
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
        self.to(device=device)
        for epoch in range(epochs):  # loop over the dataset multiple times
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
                if i % 500 == 499:  # print every 500 mini-batches
                    print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}")
                    running_loss = 0.0

        if save_to is not None:
            torch.save(self.state_dict(), save_to)
            logger.info(f"Saved model to {save_to}")

    def eval_loop(self, dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]], device: torch.device) -> float:
        """Evaluate the model."""
        self.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data in dataloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = self(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        logger.info(f"Accuracy of the network on the {total} test images: {accuracy:.2f}%")
        return accuracy

def main():
    from torch.utils.data import DataLoader
    from torchvision.datasets import CIFAR10

    from hw2 import PROJECT_ROOT
    from hw2.util import find_mean_and_stddev

    logging.basicConfig(level=logging.INFO)
    logger.info("Logging started")
    model = CNN(channels=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    cifar10_normalization = find_mean_and_stddev(CIFAR10(root=PROJECT_ROOT / "data", train=True, transform=None))

    train_loader = DataLoader(CIFAR10(root=PROJECT_ROOT / "data", train=True, transform=transform), batch_size=4, shuffle=True, num_workers=2)
    model.train_loop(criterion, optimizer, train_loader)
    model.eval()
    model.eval_loop(DataLoader(cifar10_test_dataset, batch_size=4, shuffle=False, num_workers=2))
    logger.info("Logging ended")

if __name__ == "__main__":
    main()
