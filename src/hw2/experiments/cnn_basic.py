import logging

import torch
import wandb
from torch import nn, optim
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2

from hw2 import PROJECT_ROOT, CIFAR10_NORMALIZATION
from hw2.models.cnn_basic import CNN
from hw2.util import find_max_batch_size, train_on_cifar10


def find_maximum_batch_size():
    model = CNN(channels=3)
    input_size = (3, 32, 32)
    device = "cuda"
    start_batch_size = 1
    max_batch_size = find_max_batch_size(model, input_size, device, start_batch_size)
    print(f"Maximum batch size: {max_batch_size}")


def varying_batch_size_on_training_performance():
    batch_sizes = [65536, 32768, 16384, 8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
    learning_rate = 0.001
    epochs = 5
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    logging.basicConfig(level=logging.INFO)

    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.RGB(),
            v2.ToDtype(torch.float32, scale=True),
            v2.RandomHorizontalFlip(),
            v2.RandomRotation(10),
            v2.Normalize(*CIFAR10_NORMALIZATION),
        ]
    )

    test_loader = DataLoader(
        CIFAR10(root=PROJECT_ROOT / "data", train=False, download=True, transform=transform),
        batch_size=65536,
        shuffle=False,
    )
    criterion = nn.CrossEntropyLoss()

    for batch_size in batch_sizes:
        model = CNN(channels=3)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

        run = wandb.init(
            project="CISC3027 hw2",
            group="experiment-varying-batch-size",
            job_type="train",
            config={
                "model": "CNN",
                "epochs": epochs,
                "optimizer": optimizer.__class__.__name__,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "device": device,
            }
        )

        train_on_cifar10(model, optimizer, criterion, transform, epochs=epochs, device=device, log_run=True, batch_size=batch_size, cifar_test_loader=test_loader)

        run.finish()


def varying_learning_rate_on_training_performance():
    learning_rates = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
    epochs = 5
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    logging.basicConfig(level=logging.INFO)

    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.RGB(),
            v2.ToDtype(torch.float32, scale=True),
            v2.RandomHorizontalFlip(),
            v2.RandomRotation(10),
            v2.Normalize(*CIFAR10_NORMALIZATION),
        ]
    )

    train_loader = DataLoader(
        CIFAR10(root=PROJECT_ROOT / "data", train=True, download=True, transform=transform),
        batch_size=64,
        shuffle=True,
    )

    test_loader = DataLoader(
        CIFAR10(root=PROJECT_ROOT / "data", train=False, download=True, transform=transform),
        batch_size=65536,
        shuffle=False,
    )
    criterion = nn.CrossEntropyLoss()

    for learning_rate in learning_rates:
        model = CNN(channels=3)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

        run = wandb.init(
            project="CISC3027 hw2",
            group="experiment-varying-learning-rate",
            job_type="train",
            config={
                "model": "CNN",
                "epochs": epochs,
                "optimizer": optimizer.__class__.__name__,
                "learning_rate": learning_rate,
                "batch_size": 64,
                "device": device,
            }
        )

        train_on_cifar10(model, optimizer, criterion, transform, epochs=epochs, device=device, log_run=True, batch_size=64, cifar_test_loader=test_loader, cifar_train_loader=train_loader)

        run.finish()


def using_different_optimizers():
    optimizers: list[type[Optimizer]] = [optim.SGD, optim.Adam, optim.AdamW, optim.RMSprop]
    epochs = 5
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    logging.basicConfig(level=logging.INFO)

    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.RGB(),
            v2.ToDtype(torch.float32, scale=True),
            v2.RandomHorizontalFlip(),
            v2.RandomRotation(10),
            v2.Normalize(*CIFAR10_NORMALIZATION),
        ]
    )

    train_loader = DataLoader(
        CIFAR10(root=PROJECT_ROOT / "data", train=True, download=True, transform=transform),
        batch_size=64,
        shuffle=True,
    )

    test_loader = DataLoader(
        CIFAR10(root=PROJECT_ROOT / "data", train=False, download=True, transform=transform),
        batch_size=65536,
        shuffle=False,
    )
    criterion = nn.CrossEntropyLoss()

    for optimizer_cls in optimizers:
        model = CNN(channels=3)
        optimizer = optimizer_cls(model.parameters())

        run = wandb.init(
            project="CISC3027 hw2",
            group="experiment-varying-optimizers",
            job_type="train",
            config={
                "model": "CNN",
                "epochs": epochs,
                "optimizer": optimizer.__class__.__name__,
                "batch_size": 64,
                "device": device,
            }
        )

        train_on_cifar10(model, optimizer, criterion, transform, epochs=epochs, device=device, log_run=True,
                         batch_size=64, cifar_test_loader=test_loader, cifar_train_loader=train_loader)

        run.finish()


def varying_epochs():
    epochs = 50
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    learning_rate = 0.001

    logging.basicConfig(level=logging.INFO)

    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.RGB(),
            v2.ToDtype(torch.float32, scale=True),
            v2.RandomHorizontalFlip(),
            v2.RandomRotation(10),
            v2.Normalize(*CIFAR10_NORMALIZATION),
        ]
    )

    train_loader = DataLoader(
        CIFAR10(root=PROJECT_ROOT / "data", train=True, download=True, transform=transform),
        batch_size=64,
        shuffle=True,
    )

    test_loader = DataLoader(
        CIFAR10(root=PROJECT_ROOT / "data", train=False, download=True, transform=transform),
        batch_size=65536,
        shuffle=False,
    )
    criterion = nn.CrossEntropyLoss()

    model = CNN(channels=3)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    run = wandb.init(
        project="CISC3027 hw2",
        group="experiment-varying-epochs",
        job_type="train",
        config={
            "model": "CNN",
            "epochs": epochs,
            "optimizer": optimizer.__class__.__name__,
            "batch_size": 64,
            "device": device,
            "learning_rate": learning_rate,
        }
    )

    train_on_cifar10(model, optimizer, criterion, transform, epochs=epochs, device=device, log_run=True,
                     batch_size=64, cifar_test_loader=test_loader, cifar_train_loader=train_loader)

    run.finish()


if __name__ == "__main__":
    # find_maximum_batch_size()
    # varying_batch_size_on_training_performance()
    # varying_learning_rate_on_training_performance()
    # using_different_optimizers()
    varying_epochs()
