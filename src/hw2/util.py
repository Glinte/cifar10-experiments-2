import inspect
import logging
import random
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import Any, Callable, Literal, overload, cast

import numpy as np
import pandas as pd
import torch
import wandb
from PIL import Image
from beartype import beartype
from pandas import DataFrame
from torch import nn
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, BatchSampler, RandomSampler, SequentialSampler
from torchmetrics.classification import BinaryROC
from torchvision import tv_tensors
from torchvision.datasets import CIFAR10, FashionMNIST, CIFAR100
from tqdm import tqdm
from wandb.apis.public.runs import Runs

from hw2 import PROJECT_ROOT
from hw2.data.cifar100_lt import CIFAR100LT

logger = logging.getLogger(__name__)


@beartype
def find_mean_and_stddev(dataset: Dataset[Any]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Find the mean and standard deviation of a dataset.

    The shape of the dataset is assumed to be (samples, channels, height, width). PIL Images are also supported."""
    # Get the number of channels from the first sample
    inputs, _labels = dataset[0]
    is_pil = False
    if isinstance(inputs, Image.Image):
        logger.warning("PIL images are supported, but you may want to convert them to float tensors for actual use.")
        inputs = np.array(inputs).transpose(2, 0, 1)
        is_pil = True
    num_channels = inputs.shape[0]

    # Initialize sums
    sum_c = torch.zeros(num_channels)
    sum_sq_c = torch.zeros(num_channels)
    total_pixels = 0

    for inputs, _labels in dataset:
        if is_pil:
            inputs = tv_tensors.Image(inputs)
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


def train_on_cifar(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    lr_scheduler: LRScheduler | None = None,
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
    batch_sampler: BatchSampler | None = None,
    n_test_samples: int = 500,
    save_to: str | Path | None = None,
    cifar_dataset: Literal["CIFAR10", "CIFAR100", "CIFAR100LT"] = "CIFAR10",
) -> tuple[float, float]:
    """
    Train a model on CIFAR-10/100.

    It is assumed that the model takes in 3x32x32 images and outputs 10 classes.

    Args:
        model: The model to train. The model is modified in-place.
        optimizer: The optimizer to use.
        criterion: The loss function.
        lr_scheduler: Learning rate scheduler to use.
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
        batch_sampler: Batch sampler to use for training. If provided, batch_size and shuffle are ignored.
        n_test_samples: Number of samples to use for validation during training. A full validation will be run at the end of training regardless of this parameter.
        save_to: Path to save the model to after training.
        cifar_dataset: Whether to use CIFAR-10 or CIFAR-100. If data loaders are provided, this parameter is ignored. LT stands for long-tailed.

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

    if cifar_dataset == "CIFAR10":
        CIFAR = CIFAR10
    elif cifar_dataset == "CIFAR100":
        CIFAR = CIFAR100
    elif cifar_dataset == "CIFAR100LT":
        CIFAR = CIFAR100LT
    else:
        raise ValueError(f"Unknown value {cifar_dataset} for cifar_dataset. Check type hint for valid values.")

    cifar_train_dataset = CIFAR(root=PROJECT_ROOT / "data", train=True, download=True, transform=transform)
    cifar_test_dataset = CIFAR(root=PROJECT_ROOT / "data", train=False, download=True, transform=transform)
    mnist_train_dataset = FashionMNIST(root=PROJECT_ROOT / "data", train=True, download=True, transform=transform)
    mnist_test_dataset = FashionMNIST(root=PROJECT_ROOT / "data", train=False, download=True, transform=transform)

    if batch_sampler is not None:
        train_sampler = batch_sampler
    else:
        if shuffle:
            train_sampler = RandomSampler(cifar_train_dataset, generator=g)
        else:
            train_sampler = SequentialSampler(cifar_train_dataset)
        train_sampler = BatchSampler(train_sampler, batch_size, drop_last=False)

    cifar_train_loader = cifar_train_loader or DataLoader(cifar_train_dataset, worker_init_fn=seed_worker, batch_sampler=train_sampler)
    cifar_test_loader = cifar_test_loader or DataLoader(cifar_test_dataset, batch_size=128, shuffle=False)
    mnist_train_loader = mnist_train_loader or DataLoader(mnist_train_dataset, batch_size=128, shuffle=False)
    mnist_test_loader = mnist_test_loader or DataLoader(mnist_test_dataset, batch_size=128, shuffle=False)

    model.to(device).train()
    validate_metrics = {}
    try:  # Catch keyboard interrupts to save the model
        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            total = 0

            for i, data in enumerate[tuple[torch.Tensor, torch.Tensor]](cifar_train_loader):
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

            running_loss /= len(cifar_train_loader)
            accuracy = correct / total

            if lr_scheduler is not None:
                # Each scheduler needs some different argument, so we need to check the type
                if isinstance(lr_scheduler, ReduceLROnPlateau):
                    lr_scheduler.step(running_loss)
                else:
                    try:
                        lr_scheduler.step()
                    except TypeError:
                        raise TypeError("This learning rate scheduler is not supported. It seem to require arguments and no special handling is implemented.")

            current_lr = lr_scheduler.get_last_lr()[0] if lr_scheduler is not None else optimizer.param_groups[0]["lr"]
            logger.info(f"Epoch {epoch}, Loss: {running_loss:.4f}, Accuracy: {accuracy:.4f}, LR: {current_lr}")
            # log_run=False to avoid double logging, n_samples=500 to speed up validation
            validate_metrics = validate_on_cifar(model, criterion, device=device, n_samples=n_test_samples, log_run=False, cifar_test_loader=cifar_test_loader, cifar_dataset=cifar_dataset)
            if open_set_prob_fn is not None:
                fpr, tpr, thresholds = validate_on_open_set(model, open_set_prob_fn=open_set_prob_fn, device=device, log_run=log_run, mnist_train_loader=mnist_train_loader, mnist_test_loader=mnist_test_loader, cifar_test_loader=cifar_test_loader)
            if log_run:
                assert wandb.run is not None
                wandb.run.log({"epoch": epoch, "train/loss": running_loss, "train/accuracy": accuracy}, commit=False)
                if lr_scheduler is not None:
                    wandb.run.log({"train/lr": current_lr}, commit=False)
                wandb.run.log({f"test/{metric}": value for metric, value in validate_metrics.items()})
    except KeyboardInterrupt:
        logger.info("Training interrupted. Saving model.")

    if save_to is not None:
        torch.save(model.state_dict(), save_to)
        logger.info(f"Saved model to {save_to}")
        if log_run:
            assert wandb.run is not None
            wandb.run.log_model(save_to)

    return validate_metrics["loss"], validate_metrics["accuracy"]


def validate_on_cifar(
    model: nn.Module,
    criterion: nn.Module,
    transform: Callable | None = None,
    device: torch.device = torch.device("cpu"),
    *,
    log_run: bool = False,
    cifar_test_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] | None = None,
    additional_metrics: list[Callable[[torch.Tensor, torch.Tensor], Any]] | None = None,
    n_samples: int = 10000,
    cifar_dataset: Literal["CIFAR10", "CIFAR100", "CIFAR100LT"] = "CIFAR10",
) -> dict[str, Any]:
    """
    Validate a model on CIFAR-10/100.

    Args:
        model: The model to validate.
        criterion: The loss function.
        transform: Transform to apply to the images.
        device: Device to validate on.
        log_run: Whether to log the run to Weights & Biases. This assumes that wandb.init() has already been called.
        additional_metrics: A list of functions that take in the (labels, model outputs) and return a metric.
        cifar_test_loader: DataLoader to use for validation. If None, a DataLoader is created from CIFAR-10.
        n_samples: Number of samples to use for validation. The testing loop will stop when this number of samples is reached.
        cifar_dataset: Whether to use CIFAR-10 or CIFAR-100. If the data loader is provided, this parameter is ignored.

    Returns:
        A dictionary containing the loss and accuracy, as well as any additional metrics.
        For each function in additional_metrics, the key is the function name and the value is the result.
    """

    if log_run and wandb.run is None:
        raise ValueError("wandb.init() must be called before validating with log_run=True")

    if transform and cifar_test_loader:
        raise ValueError("transform and cifar_train_loader cannot be used together")

    if cifar_dataset == "CIFAR10":
        CIFAR = CIFAR10
    elif cifar_dataset == "CIFAR100":
        CIFAR = CIFAR100
    elif cifar_dataset == "CIFAR100LT":
        CIFAR = CIFAR100LT
    else:
        raise ValueError(f"Unknown value {cifar_dataset} for cifar_dataset. Check type hint for valid values.")

    if cifar_test_loader is None:
        test_loader = DataLoader(
            CIFAR(root=PROJECT_ROOT / "data", train=False, download=True, transform=transform),
            batch_size=n_samples,
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
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs: torch.Tensor = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            predicted = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if total >= n_samples:
                break

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


@overload
def validate_on_open_set(
    model: nn.Module,
    *,
    open_set_prob_fn: Callable[[torch.Tensor], torch.Tensor],
    open_set_prob_fns: None = ...,
    transform: Callable | None = ...,
    device: torch.device = ...,
    log_run: bool = ...,
    thresholds: None | int | list[float] | torch.Tensor = ...,
    mnist_train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] | None = ...,
    mnist_test_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] | None = ...,
    cifar_test_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] | None = ...,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ...


@overload
def validate_on_open_set(
    model: nn.Module,
    *,
    open_set_prob_fn: None = ...,
    open_set_prob_fns: Sequence[Callable[[torch.Tensor], torch.Tensor]],
    transform: Callable | None = ...,
    device: torch.device = ...,
    log_run: bool = ...,
    thresholds: None | int | list[float] | torch.Tensor = ...,
    mnist_train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] | None = ...,
    mnist_test_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] | None = ...,
    cifar_test_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] | None = ...,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    ...


def validate_on_open_set(
    model: nn.Module,
    *,
    open_set_prob_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    open_set_prob_fns: Sequence[Callable[[torch.Tensor], torch.Tensor]] | None = None,
    transform: Callable | None = None,
    device: torch.device = torch.device("cpu"),
    log_run: bool = False,
    thresholds: None | int | list[float] | torch.Tensor = None,
    mnist_train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] | None = None,
    mnist_test_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] | None = None,
    cifar_test_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Using a model trained on CIFAR-10/100, validate the open-set classification on Fashion-MNIST.

    Both the training and testing dataset of Fashion-MNIST are used to validate the model.

    Args:
        model: The model to validate.
        open_set_prob_fn: A function that takes in the model outputs and calculates the probability of the input being in the open set. This function should be batch-compatible.
        open_set_prob_fns: A sequence of functions that take in the model outputs and calculate the probability of the input being in the open set. These functions should be batch-compatible.
        transform: Transform to apply to the images.
        device: Device to validate on.
        log_run: Whether to log the run to Weights & Biases. This assumes that wandb.init() has already been called.
        thresholds: Refer to torchmetrics.classification.BinaryROC for more information.
        mnist_train_loader: DataLoader of Fashion-MNIST training data. If None, a DataLoader is created from Fashion-MNIST.
        mnist_test_loader: DataLoader of Fashion-MNIST test data. If None, a DataLoader is created from Fashion-MNIST.
        cifar_test_loader: DataLoader of CIFAR test data. If None, a DataLoader is created from CIFAR-10.

    Returns:
        A list of length n_fns containing the results of the open set detection.
        Each result is a tuple of 3 tensors containing
        - fpr: A 1d tensor of size (n_thresholds, ) with false positive rate values
        - tpr: A 1d tensor of size (n_thresholds, ) with true positive rate values
        - thresholds: A 1d tensor of size (n_thresholds, ) with decreasing threshold values
    """

    if log_run and wandb.run is None:
        raise ValueError("wandb.init() must be called before validating with log_run=True")

    if open_set_prob_fn and open_set_prob_fns:
        raise ValueError("open_set_prob_fn and open_set_prob_fns cannot be used together")

    if open_set_prob_fn is None and open_set_prob_fns is None:
        raise ValueError("open_set_prob_fn or open_set_prob_fns must be provided")

    if open_set_prob_fn:
        open_set_prob_fns = [open_set_prob_fn]

    assert isinstance(open_set_prob_fns, Sequence), "open_set_prob_fns should be defined regardless of parameters at this point"

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

    cifar_test_loader = cifar_test_loader or DataLoader(
        CIFAR10(root=PROJECT_ROOT / "data", train=False, download=True, transform=transform),
        batch_size=128,
        shuffle=False,
    )

    # Cannot use train data for open set detection

    model.to(device).eval()

    # Shape: (n_fns, n_samples, n_classes)
    all_open_set_probs = torch.empty(0, device="cpu")  # Unknown size of data loader
    # Shape: (n_samples, )
    labels = torch.empty(0, device="cpu", dtype=torch.bool)

    def _load_data(data_loader, label: Literal[0, 1]):
        """Load the data and calculate the open set probabilities."""
        nonlocal all_open_set_probs, labels
        for inputs, _ in data_loader:
            inputs = inputs.to(device)

            outputs: torch.Tensor = model(inputs)
            batch_probs = torch.empty((len(open_set_prob_fns), inputs.size(0)), device="cpu")

            for j, prob_fn in enumerate(open_set_prob_fns):
                open_set_probs = prob_fn(outputs)
                batch_probs[j, :] = open_set_probs

            all_open_set_probs = torch.cat((all_open_set_probs, batch_probs), dim=1)

            if label == 0:
                labels = torch.cat((labels, torch.zeros(inputs.size(0), dtype=torch.bool, device="cpu")))
            else:
                labels = torch.cat((labels, torch.ones(inputs.size(0), dtype=torch.bool, device="cpu")))

    with torch.no_grad():
        _load_data(mnist_train_loader, label=1)
        _load_data(mnist_test_loader, label=1)
        _load_data(cifar_test_loader, label=0)

    results = []
    for i, open_set_probs in enumerate(all_open_set_probs):
        fpr, tpr, thresholds = BinaryROC(thresholds=thresholds)(open_set_probs, labels)
        results.append((fpr, tpr, thresholds))
    # if log_run:
    #     wandb.log({"roc": wandb.plot.roc_curve(all_open_set_probs, labels)})  # Duplicated effort but I don't care at this point

    if open_set_prob_fn:
        return results[0]  # Return a single result if only one function was used
    else:
        return results


def get_histories_with_config(runs: Runs) -> DataFrame:
    """Get a DataFrame of run histories with the config included."""

    histories = cast(DataFrame, runs.histories(format="pandas"))
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


def plot_per_class_accuracy(
    models_dict: dict[str, nn.Module],
    dataloader: DataLoader,
    labelnames: Sequence[Any],
    img_num_per_cls: Sequence[int],
    n_classes: int = 100,
    device: torch.device = torch.device("cpu")
):
    # Adapted from https://github.com/ShadeAlsha/LTR-weight-balancing/blob/master/utils/plot_funcs.py
    result_dict = {}
    for label in models_dict:
        model = models_dict[label]
        acc_per_class = get_per_class_acc(model, dataloader, n_classes=n_classes, device=device)
        result_dict[label] = acc_per_class

    plt.figure(figsize=(15, 4), dpi=64, facecolor='w', edgecolor='k')
    plt.xticks(list(range(100)), labelnames, rotation=90, fontsize=8);  # Set text labels.
    plt.title('per-class accuracy vs. per-class #images', fontsize=20)
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    for label in result_dict:
        ax1.bar(list(range(100)), result_dict[label], alpha=0.7, width=1, label=label, edgecolor="black")

    ax1.set_ylabel('accuracy', fontsize=16, color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue', labelsize=16)

    ax2.set_ylabel('#images', fontsize=16, color='r')
    ax2.plot(img_num_per_cls, linewidth=4, color='r')
    ax2.tick_params(axis='y', labelcolor='r', labelsize=16)

    ax1.legend(prop={'size': 14})


def get_per_class_acc(model: nn.Module, dataloader: DataLoader, n_classes: int = 100, device: torch.device = torch.device("cpu")):
    predList = np.array([])
    grndList = np.array([])
    model.eval()
    for sample in dataloader:
        with torch.no_grad():
            images, labels = sample
            images = images.to(device)
            labels = labels.type(torch.long).view(-1).numpy()
            logits = model(images)
            softmaxScores = F.softmax(logits, dim=1)

            predLabels = softmaxScores.argmax(dim=1).detach().squeeze().cpu().numpy()
            predList = np.concatenate((predList, predLabels))
            grndList = np.concatenate((grndList, labels))

    confMat = sklearn.metrics.confusion_matrix(grndList, predList)

    # normalize the confusion matrix
    a = confMat.sum(axis=1).reshape((-1, 1))
    confMat = confMat / a

    acc_avgClass = 0
    for i in range(confMat.shape[0]):
        acc_avgClass += confMat[i, i]

    acc_avgClass /= confMat.shape[0]

    acc_per_class = [0] * n_classes

    for i in range(n_classes):
        acc_per_class[i] = confMat[i, i]

    return acc_per_class
