from functools import partial
from types import MethodType
from typing import Callable
import logging

import torch
import torchvision
from torchvision.models import Inception3

from sklearn import metrics
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import BatchSampler, WeightedRandomSampler
from torchvision.transforms import transforms
import wandb

from hw2 import PROJECT_ROOT
from hw2.data.cifar100_lt import CIFAR100LT
from hw2.utils import train_on_cifar, validate_on_cifar


def get_inception_v3(num_classes: int = 1000) -> torchvision.models.Inception3:
    """
    Get an InceptionV3 model pretrained on ImageNet.

    If num_classes is not 1000, the classifier is replaced with a new one with num_classes output features.
    """
    model = torchvision.models.inception_v3(weights=torchvision.models.Inception_V3_Weights.IMAGENET1K_V1)
    if num_classes != 1000:  # Keep the original classifier and weights if num_classes is 1000
        model.fc = torch.nn.Linear(in_features=model.fc.in_features, out_features=num_classes)

    original_forward = model.forward

    def custom_forward(self, x: torch.Tensor) -> torch.Tensor:
        """InceptionV3 forward pass returns a namedtuple, we only want the output tensor."""
        # Call the original forward method
        outputs = original_forward(x)
        # Return only the logits from the original outputs
        return outputs.logits

    # Override the forward method with the custom one
    model.forward = MethodType(custom_forward, model)

    return model


def get_inception_v3_transform() -> Callable:
    """Get the transform for InceptionV3."""
    return transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, force=True)

    # Test the model and transform
    model = get_inception_v3(num_classes=10)
    transform = get_inception_v3_transform()

    config = {
        "model_type": "CNN",
        "_model_auto": model.__class__.__name__,
        "model": "Inception v3",
        "epochs": 40,
        "learning_rate": 0.001,
        "batch_size": 64,
        "device": torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        "dataset": "CIFAR100LT"
    }

    tags = ["weighted"]

    dataset = CIFAR100LT(root=PROJECT_ROOT / "data", train=True, imb_type='exp', imb_factor=0.01, download=True,
                         transform=transform)

    if config["dataset"] == "CIFAR100LT":
        class_weights = 1 / torch.tensor(dataset.img_num_per_cls, device=config["device"])
        sample_weights = [class_weights[target] for target in dataset.targets]
        sampler = BatchSampler(
            WeightedRandomSampler(weights=sample_weights, num_samples=len(dataset), replacement=True), batch_size=64,
            drop_last=False)
    else:
        class_weights = None
        sampler = None

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), weight_decay=0.1, lr=config["learning_rate"])
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"], eta_min=0)

    config.update({
        "optimizer": optimizer.__class__.__name__,
        "lr_scheduler": scheduler.__class__.__name__,
    })

    run = wandb.init(
        project="CISC3027 hw3",
        group=None,
        job_type="train",
        config=config,
        dir=PROJECT_ROOT / "wandb",
        tags=tags,
    )

    logging.basicConfig(level=logging.INFO, force=True)

    # train
    train_on_cifar(model, optimizer, criterion, scheduler, transform, config["epochs"], config["device"], log_run=True,
                   cifar_dataset=config["dataset"], n_test_samples=10000, batch_sampler=sampler)

    results = validate_on_cifar(model, criterion, CIFAR100LT.basic_test_transform, config["device"],
                                additional_metrics=[partial(metrics.classification_report, digits=4, zero_division=0)],
                                cifar_dataset="CIFAR100LT")

    torch.save(model.state_dict(), PROJECT_ROOT / "models" / "cifar100lt" / f"inceptionv3_{results["accuracy"]}.pth")