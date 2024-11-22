import logging

import torch
import torchvision
import wandb
from torch import nn
from torchvision.transforms import v2

from hw2 import PROJECT_ROOT
from hw2.open_set import one_minus_max_of_softmax
from hw2.util import CIFAR10_NORMALIZATION, train_on_cifar10


def main():
    logging.basicConfig(level=logging.INFO)
    epochs = 10
    learning_rate = 0.001
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = torchvision.models.resnet50()
    # model = torchvision.models.resnet18()
    model.fc = nn.Linear(in_features=model.fc.in_features, out_features=10)

    transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(*CIFAR10_NORMALIZATION),
    ])

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    open_set_prob_fn = one_minus_max_of_softmax

    run = wandb.init(
        project="CISC3027 hw2",
        job_type="train",
        config={
            "model": "ResNet50",  # Remember to change this to ResNet18 if you are using ResNet18
            "epochs": epochs,
            "batch_size": 64,
            "learning_rate": learning_rate,
            "optimizer": optimizer.__class__.__name__,
        }
    )

    loss, acc = train_on_cifar10(model, optimizer, criterion, transforms, epochs, device=device, open_set_prob_fn=open_set_prob_fn, log_run=True)
    torch.save(model.state_dict(), PROJECT_ROOT / f"models/resnet500_epoch{epochs}_{acc:.4f}.pth")

    # model.load_state_dict(torch.load(PROJECT_ROOT / "models/resnet18_epoch10_0.7329.pth"))
    # results = validate_on_cifar10(model, criterion, transforms, device, additional_metrics=[partial(metrics.classification_report, digits=4)])
    # print(results["classification_report"])

    run.finish()

if __name__ == "__main__":
    main()
