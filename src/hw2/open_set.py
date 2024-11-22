"""Functions to estimate the open set probability of a model."""
import torch


def one_minus_max_of_softmax(x: torch.Tensor) -> torch.Tensor:
    """Compute one minus the maximum of the softmax probabilities."""
    return x.softmax(dim=1).max(dim=1).values


def entropy_of_softmax(x: torch.Tensor, n_classes: int = 10) -> torch.Tensor:
    """Compute the entropy of the softmax probabilities, normalized by the maximum possible entropy."""
    max_entropy = torch.log(torch.tensor(n_classes).float())
    print(max_entropy)
    entropy = -(x.softmax(dim=1) * x.log_softmax(dim=1)).sum(dim=1)
    return entropy / max_entropy
