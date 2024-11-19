from __future__ import annotations

from typing import Sequence, Any, Annotated, Literal

import matplotlib as mpl
import numpy as np
from PIL import Image
from beartype import beartype
from beartype.vale import Is
from jaxtyping import Num, Integer
from matplotlib import colormaps
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from hw2.types import Array


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


@beartype
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


@beartype
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


@beartype
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


@beartype
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


@beartype
def PCA_visualization(
    X: Num[Array, "samples features"],
    y: Integer[Array, "samples"],
    target_names: Sequence[str] | None = None,
    n_components: Literal[2, 3] = 2,
    show_fig: bool = True,
) -> Figure:
    """Visualize the data using n-dimensional PCA.

    Args:
        X: The data to visualize.
        y: The labels of the data.
        target_names: The names of the labels.
        n_components: The number of components to reduce the data to, either 2 or 3.
        show_fig: Whether to display the figure.

    Returns:
        Figure: The matplotlib figure.
    """

    if n_components not in (2, 3):
        raise ValueError("Number of components must be 2 or 3.")

    # To show an interactive plot, we need to close all existing plots and switch to a different backend
    if show_fig:
        plt.close("all")
        mpl.use("Qt5Agg")

    pca = PCA(n_components=n_components)
    X_r = pca.fit_transform(X)

    fig = plt.figure()
    alpha = 1 if X.shape[0] < 1000 else 0.5
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
                alpha=alpha,
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
                alpha=alpha,
            )

    ax.legend(loc="best", shadow=False, scatterpoints=1)
    ax.set_title("PCA of CIFAR-10 dataset")
    if show_fig:
        plt.show(block=True)

    return fig


@beartype
def tsne_visualization(
    X: Num[Array, "samples features"],
    y: Integer[Array, "samples"],
    target_names: Sequence[str] | None = None,
    show_fig: bool = True,
) -> Figure:
    """
    Visualize the data using t-SNE.

    Args:
        X: The data to visualize.
        y: The labels of the data.
        target_names: The names of the labels.
        show_fig: Whether to display the figure.

    Returns:
        Figure: The matplotlib figure.
    """

    X_reduced = PCA(n_components=50).fit_transform(X)
    X_embedded = TSNE(max_iter=5000, min_grad_norm=3e-5, n_jobs=-1).fit_transform(X_reduced)

    colors = generate_distinct_colors(len(np.unique(y)))

    fig, ax = plt.subplots(figsize=(10, 10))
    for i, target_name in enumerate(target_names):
        ax.scatter(
            X_embedded[y == i, 0],
            X_embedded[y == i, 1],
            s=10,
            color=colors[i],
            label=target_name,
            alpha=1,
        )
    ax.legend(loc="best", shadow=False, scatterpoints=1)
    fig.suptitle("t-SNE")
    ax.set_title("n_iter=5000, min_grad_norm=3e-5")
    if show_fig:
        plt.show(block=True)

    return fig


def main():
    PCA_visualization(np.random.rand(1000, 3072), np.random.randint(0, 10, 1000), [f"Class {i}" for i in range(10)], n_components=3)
    # tsne_visualization(np.random.rand(1000, 3072), np.random.randint(0, 10, 1000), [f"Class {i}" for i in range(10)])

if __name__ == "__main__":
    main()
