"""Tests of the visualization module."""
import pytest
from torch.utils.data import DataLoader

from hw2.visualization import generate_distinct_colors, PCA_visualization, tsne_visualization


@pytest.mark.parametrize("n_colors", [1, 5, 10, 20, 50])
def test_generate_distinct_colors(n_colors):
    colors = generate_distinct_colors(n_colors)
    assert len(colors) == n_colors
    for color in colors:
        assert len(color) == 4  # RGBA
        assert all(0 <= c <= 1 for c in color)


@pytest.mark.mpl_image_compare
def test_PCA_visualization(dataset):
    """Test the PCA_visualization function."""

    dataloader = DataLoader(dataset, batch_size=65536, shuffle=False, num_workers=2)
    X, y = next(iter(dataloader))
    X = X.reshape(X.shape[0], -1)
    return PCA_visualization(X, y, dataset.classes, show_fig=False)


@pytest.mark.mp_image_compare
def test_tsne_visualization(dataset):
    """Test the tsne_visualization function."""

    # reduce the number of samples to speed up the test
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=2)
    X, y = next(iter(dataloader))
    X = X.reshape(X.shape[0], -1)
    return tsne_visualization(X, y, dataset.classes, show_fig=False)

