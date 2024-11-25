import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
CIFAR100LT_NORMALIZATION = ((0.5232, 0.494, 0.4442), (0.2681, 0.2612, 0.2779))
"""Normalization values for CIFAR-100 Long Tail calculated from the training dataset. First tuple is mean, second tuple is standard deviation."""
CIFAR10_NORMALIZATION = ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
"""Normalization values for CIFAR-10 calculated from the training dataset. First tuple is mean, second tuple is standard deviation."""
