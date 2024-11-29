"""Experiments related to unbalanced datasets."""
import torch
import wandb
from torch import nn, optim
from torch.optim import lr_scheduler

from hw2.data.cifar100_lt import CIFAR100LT
from hw2.models.cnn_basic import LeNet5
from hw2.util import train_on_cifar


def weighted_vs_unweighted_loss(model: nn.Module):
    """Train a model on CIFAR-100LT with and without class weights."""

    epochs = 20
    learning_rate = 0.001
    optimizer = optim.AdamW(model.parameters())
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    config = {
        "model": "CNN",
        "dataset": "CIFAR-100LT",
        "epochs": epochs,
        "optimizer": optimizer.__class__.__name__,
        "lr_scheduler": scheduler.__class__.__name__,
        "learning_rate": learning_rate,
        "batch_size": 64,
        "device": device,
    }

    run = wandb.init(
        project="CISC3027 hw3",
        group=None,
        job_type="train",
        config=config | {"weighted_loss": True},
    )
    weights = 1 / torch.tensor(CIFAR100LT().img_num_per_cls)
    train_on_cifar(model, optimizer, nn.CrossEntropyLoss(weight=weights), scheduler, CIFAR100LT.basic_train_transform, epochs, device, log_run=True, cifar_dataset="100LT")
    run.finish()

    run = wandb.init(
        project="CISC3027 hw3",
        group=None,
        job_type="train",
        config=config | {"weighted_loss": False},
    )
    train_on_cifar(model, optimizer, nn.CrossEntropyLoss(), scheduler, CIFAR100LT.basic_train_transform, epochs, device, log_run=True, cifar_dataset="100LT")
    run.finish()


if __name__ == "__main__":
    model = LeNet5(channels=3, n_classes=100)
    weighted_vs_unweighted_loss(model)
