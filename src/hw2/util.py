import inspect
import logging
import random
from functools import partial
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd
import torch
import wandb
from pandas import DataFrame
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchmetrics.classification import BinaryROC
from torchvision.datasets import CIFAR10, FashionMNIST
from tqdm import tqdm
from wandb.apis.public import Runs

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
    transform: Callable | None = None,
    epochs: int = 5,
    device: torch.device = torch.device("cpu"),
    open_set_prob_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    *,
    log_run: bool = False,
    cifar_train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] | None = None,
    cifar_test_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] | None = None,
    mnist_train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] | None = None,
    mnist_test_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] | None = None,
    batch_size: int = 64,
    shuffle: bool = True,
    seed: int | None = None,
    save_to: str | Path | None = None,
) -> tuple[float, float]:
    """
    Train a model on CIFAR-10.

    It is assumed that the model takes in 3x32x32 images and outputs 10 classes.

    Args:
        model: The model to train. The model is modified in-place.
        optimizer: The optimizer to use.
        criterion: The loss function.
        transform: Transform to apply to the images.
        epochs: Number of epochs to train.
        device: Device to train on.
        open_set_prob_fn: A function that takes in the model outputs and calculates the probability of the input being in the open set. This function should be batch-compatible. If None, open set detection is not performed.
        log_run: Whether to log the run to Weights & Biases. This assumes that wandb.init() has already been called.
        cifar_train_loader: DataLoader to use for training. If None, a DataLoader is created from CIFAR-10.
        cifar_test_loader: DataLoader to use for validation. If None, a DataLoader is created from CIFAR-10.
        mnist_train_loader: DataLoader to use for training. If None, a DataLoader is created from Fashion-MNIST.
        mnist_test_loader: DataLoader to use for validation. If None, a DataLoader is created from Fashion-MNIST.
        batch_size: Batch size for training.
        shuffle: Whether to shuffle the training data.
        seed: Seed for reproducibility.
        save_to: Path to save the model to after training.

    Returns:
        Loss and accuracy on the test set after training.
    """

    if log_run and wandb.run is None:
        raise ValueError("wandb.init() must be called before training with log_run=True")

    if transform and (cifar_train_loader or cifar_test_loader):
        logger.warning("transform is provided but cifar_train_loader or cifar_test_loader is also provided. Ignoring transform.")

    def seed_worker(worker_id: int) -> None:
        """Seed the worker RNG for reproducibility."""
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    if seed is not None:
        g.manual_seed(seed)

    if cifar_train_loader is None:
        train_loader = DataLoader(
            CIFAR10(root=PROJECT_ROOT / "data", train=True, download=True, transform=transform),
            batch_size=batch_size,
            shuffle=shuffle,
            worker_init_fn=seed_worker,
            generator=g,
        )
    else:
        train_loader = cifar_train_loader

    if cifar_test_loader is None:
        test_loader = DataLoader(
            CIFAR10(root=PROJECT_ROOT / "data", train=False, download=True, transform=transform),
            batch_size=batch_size,
            shuffle=False,
        )
    else:
        test_loader = cifar_test_loader

    mnist_train_loader = mnist_train_loader or DataLoader(
        FashionMNIST(root=PROJECT_ROOT / "data", train=True, download=True, transform=transform),
        batch_size=128,
        shuffle=False,
    )

    mnist_test_loader = mnist_test_loader or DataLoader(
        FashionMNIST(root=PROJECT_ROOT / "data", train=False, download=True, transform=transform),
        batch_size=128,
        shuffle=False,
    )

    model.to(device).train()
    validate_metrics = None
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

            if batch_size > 500 or i % (500 // batch_size) == 0:
                logger.info(f"Epoch {epoch}, Batch {i}, Loss: {loss.item():.4f}")

        running_loss /= len(train_loader)
        accuracy = correct / total

        logger.info(f"Epoch {epoch}, Loss: {running_loss:.4f}, Accuracy: {accuracy:.4f}")
        validate_metrics = validate_on_cifar10(model, criterion, device=device, log_run=False, cifar_test_loader=test_loader)  # log_run=False to avoid double logging
        if open_set_prob_fn is not None:
            fpr, tpr, thresholds = validate_on_open_set(model, open_set_prob_fn=open_set_prob_fn, device=device, log_run=log_run, mnist_train_loader=mnist_train_loader, mnist_test_loader=mnist_test_loader, cifar10_test_loader=test_loader)
        if log_run:
            wandb.run.log({"epoch": epoch, "train/loss": running_loss, "train/accuracy": accuracy}, commit=False)
            wandb.run.log({f"test/{metric}": value for metric, value in validate_metrics.items()})

    if save_to is not None:
        torch.save(model.state_dict(), save_to)
        logger.info(f"Saved model to {save_to}")
        if log_run:
            wandb.run.log_model(save_to)

    return validate_metrics["loss"], validate_metrics["accuracy"]


def validate_on_cifar10(
    model: nn.Module,
    criterion: nn.Module,
    transform: Callable | None = None,
    device: torch.device = torch.device("cpu"),
    *,
    log_run: bool = False,
    cifar_test_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] | None = None,
    additional_metrics: list[Callable[[torch.Tensor, torch.Tensor], Any]] | None = None,
) -> dict[str, Any]:
    """
    Validate a model on CIFAR-10.

    Args:
        model: The model to validate.
        criterion: The loss function.
        transform: Transform to apply to the images.
        device: Device to validate on.
        log_run: Whether to log the run to Weights & Biases. This assumes that wandb.init() has already been called.
        additional_metrics: A list of functions that take in the (labels, model outputs) and return a metric.
        cifar_test_loader: DataLoader to use for validation. If None, a DataLoader is created from CIFAR-10.

    Returns:
        A dictionary containing the loss and accuracy, as well as any additional metrics.
        For each function in additional_metrics, the key is the function name and the value is the result.
    """

    if log_run and wandb.run is None:
        raise ValueError("wandb.init() must be called before validating with log_run=True")

    if transform and cifar_test_loader:
        raise ValueError("transform and cifar_train_loader cannot be used together")

    if cifar_test_loader is None:
        test_loader = DataLoader(
            CIFAR10(root=PROJECT_ROOT / "data", train=False, download=True, transform=transform),
            batch_size=65536,
            shuffle=False,
        )
    else:
        test_loader = cifar_test_loader

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
    labels, outputs = labels.cpu(), outputs.cpu()
    for metric in additional_metrics:
        if inspect.isfunction(metric):
            metric_name = metric.__name__
        elif isinstance(metric, partial):
            metric_name = metric.func.__name__
        else:
            raise ValueError("additional_metrics must be a list of functions or partials, or otherwise have a __name__ attribute")

        if metric_name == "classification_report":  # Special case for classification_report
            metrics_dict[metric_name] = metric(labels, outputs.argmax(dim=1))
        else:
            metrics_dict[metric_name] = metric(labels, outputs)

    logger.info(f"Validation {", ".join(f'{metric}: {value}' for metric, value in metrics_dict.items())}")
    if log_run:
        wandb.run.log({f"test/{metric}": value for metric, value in metrics_dict.items()})

    return metrics_dict


def validate_on_open_set(
    model: nn.Module,
    open_set_prob_fn: Callable[[torch.Tensor], torch.Tensor],
    transform: Callable | None = None,
    device: torch.device = torch.device("cpu"),
    *,
    log_run: bool = False,
    thresholds: None | int | list[float] | torch.Tensor = None,
    mnist_train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] | None = None,
    mnist_test_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] | None = None,
    cifar10_test_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Using a model trained on CIFAR-10, validate the open-set classification on Fashion-MNIST.

    Both the training and testing dataset of Fashion-MNIST are used to validate the model.

    Args:
        model: The model to validate.
        open_set_prob_fn: A function that takes in the model outputs and calculates the probability of the input being in the open set. This function should be batch-compatible.
        transform: Transform to apply to the images.
        device: Device to validate on.
        log_run: Whether to log the run to Weights & Biases. This assumes that wandb.init() has already been called.
        thresholds: Refer to torchmetrics.classification.BinaryROC for more information.
        mnist_train_loader: DataLoader of Fashion-MNIST training data. If None, a DataLoader is created from Fashion-MNIST.
        mnist_test_loader: DataLoader of Fashion-MNIST test data. If None, a DataLoader is created from Fashion-MNIST.
        cifar10_test_loader: DataLoader of CIFAR-10 test data. If None, a DataLoader is created from CIFAR-10.

    Returns:
        A tuple of 3 tensors containing
        - fpr: A 1d tensor of size (n_thresholds, ) with false positive rate values
        - tpr: A 1d tensor of size (n_thresholds, ) with true positive rate values
        - thresholds: A 1d tensor of size (n_thresholds, ) with decreasing threshold values
    """

    if log_run and wandb.run is None:
        raise ValueError("wandb.init() must be called before validating with log_run=True")

    mnist_train_loader = mnist_train_loader or DataLoader(
        FashionMNIST(root=PROJECT_ROOT / "data", train=True, download=True, transform=transform),
        batch_size=65536,
        shuffle=False,
    )

    mnist_test_loader = mnist_test_loader or DataLoader(
        FashionMNIST(root=PROJECT_ROOT / "data", train=False, download=True, transform=transform),
        batch_size=128,
        shuffle=False,
    )

    cifar10_test_loader = cifar10_test_loader or DataLoader(
        CIFAR10(root=PROJECT_ROOT / "data", train=False, download=True, transform=transform),
        batch_size=128,
        shuffle=False,
    )

    # Cannot use train data for open set detection

    model.to(device).eval()

    all_open_set_probs = torch.empty(0, device="cpu")
    labels = torch.empty(0, device="cpu", dtype=torch.bool)

    def _load_data(data_loader, label: Literal[0, 1]):
        """Load the data and calculate the open set probabilities."""
        nonlocal all_open_set_probs, labels
        for i, data in enumerate[tuple[torch.Tensor, torch.Tensor]](data_loader):
            inputs, _ = data
            inputs = inputs.to(device)

            outputs: torch.Tensor = model(inputs)
            open_set_prob = open_set_prob_fn(outputs.to("cpu"))

            all_open_set_probs = torch.cat((all_open_set_probs, open_set_prob))
            if label == 0:
                labels = torch.cat((labels, torch.zeros_like(open_set_prob, dtype=torch.bool)))
            else:
                labels = torch.cat((labels, torch.ones_like(open_set_prob, dtype=torch.bool)))

    with torch.no_grad():
        _load_data(mnist_train_loader, label=1)
        _load_data(mnist_test_loader, label=1)
        _load_data(cifar10_test_loader, label=0)

    fpr, tpr, thresholds = BinaryROC(thresholds=thresholds)(all_open_set_probs, labels)
    if log_run:
        wandb.log({"roc": wandb.plot.roc_curve(all_open_set_probs, labels)})  # Duplicated effort but I don't care at this point

    return fpr, tpr, thresholds


def get_histories_with_config(runs: Runs) -> DataFrame:
    """Get a DataFrame of run histories with the config included."""

    histories = runs.histories(format="pandas")
    configs = []

    for run in tqdm(runs):
        run_config = run.config
        run_config_df = DataFrame.from_dict(run_config, orient='index').T
        run_config_df["run_id"] = run.id
        run_config_df["run_name"] = run.name
        configs.append(run_config_df)

    config_df = pd.concat(configs, axis=0)
    assert isinstance(config_df, DataFrame)
    histories = pd.merge(config_df, histories, left_on="run_id", right_on="run_id")
    return histories
