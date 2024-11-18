import pytest
import torch
from sklearn import metrics
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms

from hw2 import PROJECT_ROOT
from hw2.models.cnn_basic import CNN

@pytest.fixture
def device():
    """Return the appropriate device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_old_cnn_model_f1_score_on_cifar_10(device) -> None:
    """Should achieve F1 score of 0.60 on CIFAR-10 test set."""
    model = CNN(channels=3)
    model.load_state_dict(torch.load(PROJECT_ROOT / "models" / "202409260908_cnn_Adam_5(60).pth", weights_only=True))
    model.to(device=device)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    cifar10_test_dataset = CIFAR10(root=PROJECT_ROOT / "data", train=False, download=True, transform=transform)

    y_true = []
    y_pred = []

    with torch.no_grad():
        for data in DataLoader(cifar10_test_dataset, batch_size=4, shuffle=False, num_workers=2):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    f1 = metrics.f1_score(y_true, y_pred, average="weighted")
    print(f"F1 score: {f1}")
    assert 0.60 < f1 < 0.61
