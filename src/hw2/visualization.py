from __future__ import annotations

import random
from typing import Sequence, TYPE_CHECKING, Any, Annotated

import numpy as np
from PIL import Image
import matplotlib as mpl
from beartype import beartype
from beartype.vale import Is
from matplotlib import pyplot as plt, cm
from matplotlib.colors import ListedColormap
from matplotlib import colormaps
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

if TYPE_CHECKING:
    import numpy.typing as npt
    from stubs.sklearn._typing import MatrixLike


@beartype
def visualize_rgb_image(
    data: Annotated[np.ndarray[Any, np.dtype[Any]], Is[lambda data: data.size == 3072]],
    *,
    show: bool = True,
) -> Image.Image:
    """Visualize a single RGB image."""

    image = Image.fromarray(data.astype(np.uint8).reshape(3, 32, 32).transpose(1, 2, 0))
    if show:
        image.show()
    return image


@beartype
def visualize_grayscale_image(
    data: Annotated[np.ndarray[Any, np.dtype[Any]], Is[lambda data: data.size == 1024]],
    *,
    show: bool = True,
) -> Image.Image:
    """Visualize a single 32x32 grayscale image."""

    image = Image.fromarray(data.reshape(32, 32))
    if show:
        image.show()
    return image


@beartype
def visualize_images(
    data: Annotated[
        np.ndarray,
        Is[lambda data: data[0].size == 1024 or data[0].size == 3072],
    ],
    *,
    show: bool = True,
    overlay: Annotated[
        np.ndarray,
        Is[lambda data: data[0].size == 1024 or data[0].size == 3072],
    ]
    | None = None,
) -> Image.Image:
    """Visualize multiple 32x32 images. Images can be either grayscale or RGB.

    Images are displayed as a square grid of images.
    
    Args:
        data: The images to visualize.
        show: Whether to display the image in a window.
        overlay: The images to overlay on top of the original images.
    """

    if data[0].size == 1024:
        images = [Image.fromarray(i.reshape(32, 32)) for i in data]
    else:
        images = [Image.fromarray(i.astype(np.uint8).reshape(3, 32, 32).transpose(1, 2, 0)) for i in data]

    total_images = len(images)
    rows = int(np.sqrt(total_images))
    cols = total_images // rows

    combined_image = combine_images_vertically(
        [combine_images_horizontally(images[i : i + cols]) for i in range(0, total_images, cols)]
    )
    if overlay is not None:
        overlay_image = visualize_images(overlay, show=False, overlay=None)
        combined_image = Image.blend(combined_image, overlay_image, alpha=0.6)
    if show:
        combined_image.show()
    return combined_image


def combine_images_horizontally(images: Sequence[Image.Image]) -> Image.Image:
    """Combine images horizontally."""
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)

    new_image = Image.new("RGB", (total_width, max_height))
    x_offset = 0
    for image in images:
        new_image.paste(image, (x_offset, 0))
        x_offset += image.size[0]

    return new_image


def combine_images_vertically(images: Sequence[Image.Image]) -> Image.Image:
    """Combine images vertically."""
    widths, heights = zip(*(i.size for i in images))
    max_width = max(widths)
    total_height = sum(heights)

    new_image = Image.new("RGB", (max_width, total_height))
    y_offset = 0
    for image in images:
        new_image.paste(image, (0, y_offset))
        y_offset += image.size[1]

    return new_image


def generate_distinct_colors(n_colors: int) -> list[tuple[float, float, float, float]]:
    """
    Generate visually distinct colors for scatter plot classes.

    Args:
        n_colors (int): Number of distinct colors needed

    Returns:
        list: List of RGBA tuples (values may be float or np.float64)
    """
    if n_colors <= 10:
        # For small number of classes, use qualitative colormap
        colormap: ListedColormap = colormaps['Set3']
        colors = [colormap(i) for i in np.linspace(0, 1, n_colors)]
    else:
        # For larger number of classes, generate random colors with good spacing
        colors = []
        for i in range(n_colors):
            # Use HSV color space for better distribution
            hue = i / n_colors
            saturation = 0.7 + np.random.rand() * 0.3  # 0.7-1.0
            value = 0.8 + np.random.rand() * 0.2  # 0.8-1.0

            # Convert HSV to RGB
            h = hue * 6
            c = value * saturation
            x = c * (1 - abs(h % 2 - 1))
            m = value - c

            if h < 1:
                rgb = (c, x, 0)
            elif h < 2:
                rgb = (x, c, 0)
            elif h < 3:
                rgb = (0, c, x)
            elif h < 4:
                rgb = (0, x, c)
            elif h < 5:
                rgb = (x, 0, c)
            else:
                rgb = (c, 0, x)

            colors.append((rgb[0] + m, rgb[1] + m, rgb[2] + m, 1))

    return colors


def display_color_palette(n_colors: int) -> None:
    """
    Display the generated colors in a horizontal strip.

    Args:
        n_colors (int): Number of colors to generate and display
    """
    colors = generate_distinct_colors(n_colors)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 2))

    # Create a rectangle for each color
    for i, color in enumerate(colors):
        ax.add_patch(plt.Rectangle((i, 0), 1, 1, facecolor=color))

    # Set the axis limits and aspect ratio
    ax.set_xlim(0, n_colors)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')

    # Remove axes and ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Add color index labels
    for i in range(n_colors):
        ax.text(i + 0.5, -0.1, str(i), ha='center', va='top')

    plt.title(f'Generated Color Palette ({n_colors} colors)')
    plt.show()


def PCA_visualization(
    X: MatrixLike,
    y: np.ndarray[Any, np.dtype[np.integer[Any]]],
    target_names: Sequence[str] | None = None,
    n_components: int = 2,
) -> None:
    """Visualize the data using n-dimensional PCA.

    Args:
        X: The data to visualize.
        y: The labels of the data.
        target_names: The names of the labels.
        n_components: The number of components to reduce the data to, either 2 or 3.
    """

    if n_components not in (2, 3):
        raise ValueError("Number of components must be 2 or 3.")

    # Set matplotlib to be interactive
    mpl.use("Qt5Agg")
    plt.ion()

    pca = PCA(n_components=n_components)
    X_r = pca.fit_transform(X)

    fig = plt.figure()
    if n_components == 2:
        ax = fig.add_subplot(111)
        colors = generate_distinct_colors(len(np.unique(y)))
        if target_names is None:
            target_names = [f"Class {i}" for i in range(len(colors))]
        for i, target_name in enumerate(target_names):
            ax.scatter(
                X_r[y == i, 0],
                X_r[y == i, 1],
                color=colors[i],
                label=target_name,
            )
    else:
        ax = fig.add_subplot(111, projection="3d")
        colors = generate_distinct_colors(len(np.unique(y)))
        if target_names is None:
            target_names = [f"Class {i}" for i in range(len(colors))]
        for i, target_name in enumerate(target_names):
            ax.scatter(
                X_r[y == i, 0],
                X_r[y == i, 1],
                X_r[y == i, 2],
                color=colors[i],
                label=target_name,
            )

    ax.legend(loc="best", shadow=False, scatterpoints=1)
    ax.set_title("PCA of CIFAR-10 dataset")
    plt.show(block=True)


def tsne_visualization(X: MatrixLike, y: np.ndarray[Any, np.dtype[np.integer[Any]]], target_names: Sequence[str] | None = None) -> None:
    pass


def main():
    # FIXME
    data_slice = slice(0, 50000)
    train_data_flat = train_data.reshape(train_data.shape[0], -1)
    train_data_flat_reduced = PCA(n_components=200).fit_transform(train_data_flat[data_slice])
    X_embedded = TSNE(verbose=2, n_iter=5000, min_grad_norm=3e-5, n_jobs=-1).fit_transform(train_data_flat_reduced)

    colors = [
        "navy",
        "turquoise",
        "darkorange",
        "red",
        "green",
        "blue",
        "purple",
        "yellow",
        "black",
        "pink",
    ]

    plt.figure(figsize=(10, 10))
    for i, target_name in enumerate(label_names):
        plt.scatter(
            X_embedded[train_labels[data_slice] == i, 0],
            X_embedded[train_labels[data_slice] == i, 1],
            s=10,
            color=colors[i],
            label=target_name,
            alpha=0.4,
        )
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.title("t-SNE of CIFAR-10 dataset")
    plt.show()

if __name__ == "__main__":
    main()
