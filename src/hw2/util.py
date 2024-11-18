from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


def find_mean_and_stddev(dataset: Dataset[Any]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Find the mean and standard deviation of a dataset.

    The shape of the dataset is assumed to be (samples, channels, height, width)."""
    # Get the number of channels from the first sample
    inputs, _labels = dataset[0]
    num_channels = inputs.shape[0]

    # Initialize sums
    sum_c = torch.zeros(num_channels)
    sum_sq_c = torch.zeros(num_channels)
    total_pixels = 0

    for inputs, _labels in dataset:
        inputs = inputs.float()
        c, h, w = inputs.shape
        pixels = h * w
        total_pixels += pixels

        sum_c += inputs.sum(dim=[1, 2])
        sum_sq_c += (inputs ** 2).sum(dim=[1, 2])

    mean_c = sum_c / total_pixels
    std_c = torch.sqrt(sum_sq_c / total_pixels - mean_c ** 2)

    mean_tuple = tuple(mean_c.tolist())
    std_tuple = tuple(std_c.tolist())

    return mean_tuple, std_tuple


CIFAR10_NORMALIZATION = ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
"""Normalization values for CIFAR-10 calculated from the training dataset. First tuple is mean, second tuple is standard deviation."""
