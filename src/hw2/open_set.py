"""Functions to estimate the open set probability of a model."""
import torch


def one_minus_max_of_prob(x: torch.Tensor, softmax: bool = True) -> torch.Tensor:
    """Compute one minus the maximum of the class probabilities.

    Args:
        x: The class probabilities or logits.
        softmax: Whether to apply softmax to x before computing.
    """
    x = torch.softmax(x, dim=1) if softmax else x
    return 1 - x.max(dim=1).values


def one_minus_sum_of_topk_prob(x: torch.Tensor, k: int = 3, softmax: bool = True) -> torch.Tensor:
    """Compute one minus the sum of the top k class probabilities.

    Args:
        x: The class probabilities or logits.
        k: The number of top classes to sum.
        softmax: Whether to apply softmax to x before computing.
    """
    x = torch.softmax(x, dim=1) if softmax else x
    return 1 - x.topk(k).values.sum(dim=1)


def entropy_of_prob(x: torch.Tensor, n_classes: int = 10, softmax: bool = True) -> torch.Tensor:
    """Compute the entropy of the class probabilities, normalized by the maximum possible entropy.

    Args:
        x: The class probabilities or logits.
        n_classes: The number of classes.
        softmax: Whether to apply softmax to x before computing.
    """
    x = torch.softmax(x, dim=1) if softmax else x
    max_entropy = torch.log(torch.tensor(n_classes).float())
    entropy = -torch.sum(x * torch.log(x), dim=1)
    return entropy / max_entropy


def diff_of_logits(x: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """Compute the difference between the maximum and second maximum logits. Normalizing by (1 - d / d + c).

    Args:
        x: The class logits.
    """
    diff = x.topk(2).values[:, 0] - x.topk(2).values[:, 1]
    return 1 - diff / (diff + c)


def energy_score(x: torch.Tensor) -> torch.Tensor:
    """Compute energy score for open set estimation."""
    return -torch.logsumexp(x, dim=1)
