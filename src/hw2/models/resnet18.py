import torch
import torchvision
import wandb
from torch import nn
from torchvision.transforms import v2

from hw2.open_set import one_minus_max_of_softmax
from hw2.util import CIFAR10_NORMALIZATION, train_on_cifar10


def validate(model, test_loader, criterion):
    model.eval()
    running_loss = 0
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            running_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
    running_loss /= len(test_loader)
    accuracy = correct / len(test_loader.dataset)
    return running_loss, accuracy


def main():
    epochs = 10
    learning_rate = 0.001
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = torchvision.models.resnet18()
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
            "model": "ResNet18",
            "epochs": epochs,
            "batch_size": 64,
            "learning_rate": learning_rate,
            "optimizer": optimizer.__class__.__name__,
        }
    )

    train_on_cifar10(model, optimizer, criterion, transforms, epochs, device=device, open_set_prob_fn=open_set_prob_fn)

    run.finish()

if __name__ == "__main__":
    main()
