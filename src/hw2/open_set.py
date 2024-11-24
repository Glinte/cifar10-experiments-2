"""Functions to estimate the open set probability of a model."""
import torch


def one_minus_max_of_prob(x: torch.Tensor) -> torch.Tensor:
    """Compute one minus the maximum of the class probabilities."""
    return 1 - x.max(dim=1).values


def entropy_of_softmax(x: torch.Tensor, n_classes: int = 10) -> torch.Tensor:
    """Compute the entropy of the class probabilities, normalized by the maximum possible entropy."""
    max_entropy = torch.log(torch.tensor(n_classes).float())
    entropy = -torch.sum(x * torch.log(x), dim=1)
    return entropy / max_entropy
