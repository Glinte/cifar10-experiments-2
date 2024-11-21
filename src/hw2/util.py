import logging
import random
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import wandb
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchmetrics.classification import BinaryROC
from torchvision.datasets import CIFAR10, FashionMNIST

from hw2 import PROJECT_ROOT

logger = logging.getLogger(__name__)


def find_mean_and_stddev(dataset: Dataset[Any]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Find the mean and standard deviation of a dataset.

    The shape of the dataset is assumed to be (samples, channels, height, width)."""
    # Get the number of channels from the first sample
    inputs, _labels = dataset[0]
    num_channels = inputs.shape[0]

    # Initialize sums
    sum_c = torch.zeros(num_channels)
    sum_sq_c = torch.zeros(num_channels)
    total_pixels = 0

    for inputs, _labels in dataset:
        inputs = inputs.float()
        c, h, w = inputs.shape
        pixels = h * w
        total_pixels += pixels

        sum_c += inputs.sum(dim=[1, 2])
        sum_sq_c += (inputs**2).sum(dim=[1, 2])

    mean_c = sum_c / total_pixels
    std_c = torch.sqrt(sum_sq_c / total_pixels - mean_c**2)

    mean_tuple = tuple(mean_c.tolist())
    std_tuple = tuple(std_c.tolist())

    return mean_tuple, std_tuple


CIFAR10_NORMALIZATION = ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
"""Normalization values for CIFAR-10 calculated from the training dataset. First tuple is mean, second tuple is standard deviation."""


def find_max_batch_size(model, input_size, device="cuda", start_batch_size=1):
    """
    Find the maximum batch size that fits in GPU memory for a given model and input size.

    Args:
        model: The PyTorch model to test.
        input_size: Tuple representing the input dimensions (C, H, W).
        device: Device to test on ("cuda" or "cpu").
        start_batch_size: Initial batch size to test.

    Returns:
        max_batch_size: The largest batch size that fits in GPU memory.
    """
    model.to(device)
    batch_size = start_batch_size
    max_batch_size = start_batch_size

    while True:
        try:
            dummy_input = torch.randn((batch_size, *input_size), device=device)
            with torch.no_grad():
                model(dummy_input)
            max_batch_size = batch_size
            batch_size *= 2
        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                break
            else:
                raise e

    return max_batch_size


def train_on_cifar10(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    data_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] | None = None,
    transform: Callable | None = None,
    epochs: int = 5,
    device: torch.device = torch.device("cpu"),
    *,
    log_run: bool = False,
    batch_size: int = 4,
    shuffle: bool = True,
    seed: int | None = None,
    save_to: str | Path | None = None,
) -> None:
    """
    Train a model on CIFAR-10.

    It is assumed that the model takes in 3x32x32 images and outputs 10 classes.

    Args:
        model: The model to train. The model is modified in-place.
        optimizer: The optimizer to use.
        criterion: The loss function.
        data_loader: DataLoader to use for training. If None, a DataLoader is created from CIFAR-10.
        transform: Transform to apply to the images.
        epochs: Number of epochs to train.
        device: Device to train on.
        log_run: Whether to log the run to Weights & Biases. This assumes that wandb.init() has already been called.
        batch_size: Batch size for training.
        shuffle: Whether to shuffle the training data.
        seed: Seed for reproducibility.
        save_to: Path to save the model to after training.
    """

    if log_run and wandb.run is None:
        raise ValueError("wandb.init() must be called before training with log_run=True")

    def seed_worker(worker_id: int) -> None:
        """Seed the worker RNG for reproducibility."""
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    if seed is not None:
        g.manual_seed(seed)

    if data_loader is None:
        train_loader = DataLoader(
            CIFAR10(root=PROJECT_ROOT / "data", train=True, download=True, transform=transform),
            batch_size=batch_size,
            shuffle=shuffle,
            worker_init_fn=seed_worker,
            generator=g,
        )
    else:
        train_loader = data_loader

    model.to(device).train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for i, data in enumerate[tuple[torch.Tensor, torch.Tensor]](train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs: torch.Tensor = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predicted = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        running_loss /= len(train_loader)
        accuracy = correct / total

        logger.info(f"Epoch {epoch}, Loss: {running_loss:.4f}, Accuracy: {accuracy:.4f}")
        validate_metrics = validate_on_cifar10(model, criterion, transform, device, log_run=False)  # log_run=False to avoid double logging
        if log_run:
            wandb.run.log({"epoch": epoch, "train/loss": running_loss, "train/accuracy": accuracy}, commit=False)
            wandb.run.log({f"test/{metric}": value for metric, value in validate_metrics.items()})

    if save_to is not None:
        torch.save(model.state_dict(), save_to)
        logger.info(f"Saved model to {save_to}")
        if log_run:
            wandb.run.log_model(save_to)


def validate_on_cifar10(
    model: nn.Module,
    criterion: nn.Module,
    data_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] | None = None,
    transform: Callable | None = None,
    device: torch.device = torch.device("cpu"),
    *,
    log_run: bool = False,
    additional_metrics: list[Callable[[torch.Tensor, torch.Tensor], Any]] | None = None,
) -> dict[str, Any]:
    """
    Validate a model on CIFAR-10.

    Args:
        model: The model to validate.
        criterion: The loss function.
        data_loader: DataLoader to use for validation. If None, a DataLoader is created from CIFAR-10.
        transform: Transform to apply to the images.
        device: Device to validate on.
        log_run: Whether to log the run to Weights & Biases. This assumes that wandb.init() has already been called.
        additional_metrics: A list of functions that take in the model outputs and labels and return a metric.

    Returns:
        A dictionary containing the loss and accuracy, as well as any additional metrics.
        For each function in additional_metrics, the key is the function name and the value is the result.
    """

    if log_run and wandb.run is None:
        raise ValueError("wandb.init() must be called before validating with log_run=True")

    if data_loader is None:
        test_loader = DataLoader(
            CIFAR10(root="data", train=False, download=True, transform=transform),
            batch_size=64,
            shuffle=False,
        )
    else:
        test_loader = data_loader

    model.to(device).eval()
    running_loss = 0.0
    correct = 0
    total = 0

    if additional_metrics is None:
        additional_metrics = []

    with torch.no_grad():
        for i, data in enumerate[tuple[torch.Tensor, torch.Tensor]](test_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs: torch.Tensor = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            predicted = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    running_loss /= len(test_loader)
    accuracy = correct / total

    metrics_dict = {"loss": running_loss, "accuracy": accuracy}
    for metric in additional_metrics:
        metrics_dict[metric.__name__] = metric(outputs, labels)

    logger.info(f"Validation {", ".join(f'{metric}: {value:.4f}' for metric, value in metrics_dict.items())}")
    if log_run:
        wandb.run.log({f"test/{metric}": value for metric, value in metrics_dict.items()})

    return metrics_dict


def validate_on_fashion_mnist(
    model: nn.Module,
    open_set_prob_fn: Callable[[torch.Tensor], torch.Tensor],
    transform: Callable | None = None,
    device: torch.device = torch.device("cpu"),
    *,
    log_run: bool = False,
    thresholds: None | int | list[float] | torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Using a model trained on CIFAR-10, validate the open-set classification on Fashion-MNIST.

    Both the training dataset of Fashion-MNIST are used to validate the model.

    Args:
        model: The model to validate.
        open_set_prob_fn: A function that takes in the model outputs and calculates the probability of the input being in the open set. This function should be batch-compatible.
        transform: Transform to apply to the images.
        device: Device to validate on.
        log_run: Whether to log the run to Weights & Biases. This assumes that wandb.init() has already been called.
        thresholds: Refer to torchmetrics.classification.BinaryROC for more information.

    Returns:
        A tuple of 3 tensors containing
        - fpr: A 1d tensor of size (n_thresholds, ) with false positive rate values
        - tpr: A 1d tensor of size (n_thresholds, ) with true positive rate values
        - thresholds: A 1d tensor of size (n_thresholds, ) with decreasing threshold values
    """

    if log_run and wandb.run is None:
        raise ValueError("wandb.init() must be called before validating with log_run=True")

    mnist_train_loader = DataLoader(
        FashionMNIST(root=PROJECT_ROOT / "data", train=True, download=True, transform=transform),
        batch_size=64,
        shuffle=False,
    )

    mnist_test_loader = DataLoader(
        FashionMNIST(root=PROJECT_ROOT / "data", train=False, download=True, transform=transform),
        batch_size=64,
        shuffle=False,
    )

    cifar10_test_loader = DataLoader(
        CIFAR10(root=PROJECT_ROOT / "data", train=False, download=True, transform=transform),
        batch_size=64,
        shuffle=False,
    )

    # Cannot use train data for open set detection

    model.to(device).eval()

    all_open_set_probs = torch.empty(0)
    labels = torch.empty(0)

    def _load_data(data_loader):
        """Load the data and calculate the open set probabilities."""
        nonlocal all_open_set_probs, labels
        for i, data in enumerate[tuple[torch.Tensor, torch.Tensor]](data_loader):
            inputs, _ = data
            inputs = inputs.to(device)

            outputs: torch.Tensor = model(inputs)
            open_set_prob = open_set_prob_fn(outputs)

            all_open_set_probs = torch.cat((all_open_set_probs, open_set_prob))
            labels = torch.cat((labels, torch.ones(open_set_prob)))

    with torch.no_grad():
        _load_data(mnist_train_loader)
        _load_data(mnist_test_loader)
        _load_data(cifar10_test_loader)

    fpr, tpr, thresholds = BinaryROC(thresholds=thresholds)(all_open_set_probs, labels)
    if log_run:
        wandb.log({"roc": wandb.plot.roc_curve(all_open_set_probs, labels)})  # Duplicated effort but I don't care at this point

    return fpr, tpr, thresholds
