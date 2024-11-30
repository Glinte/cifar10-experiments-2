from __future__ import annotations

from typing import Sequence, Any, Annotated, Literal

import matplotlib as mpl
import numpy as np
from PIL import Image
from beartype import beartype
from beartype.door import is_bearable
from beartype.vale import Is
from hw2 import PROJECT_ROOT
from jaxtyping import Num, Integer, Float, jaxtyped
from matplotlib import colormaps
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch

from hw2.types_ import Array
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader


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


def _convert_to_images(data: Float[Array, "N ..."]) -> Sequence[Image.Image]:
    """Convert an array of 32x32 images to a list of PIL images."""
    data = np.array(data)
    if data[0].size == 1024:
        images = [Image.fromarray(img.reshape(32, 32)) for img in data]
    else:
        images = [Image.fromarray(img.reshape(3, 32, 32).astype(np.uint8).transpose(1, 2, 0)) for img in data]
    return images


@jaxtyped(typechecker=beartype)
def visualize_images(
    data: Float[Array, "N *Img"] | Sequence[Image.Image],
    *,
    show: bool = True,
    overlay: Float[Array, "N *Img"] | Sequence[Image.Image] | None = None,
) -> Image.Image:
    """Visualize multiple 32x32 images. Images can be either grayscale or RGB.

    Images are displayed as a square grid of images.
    
    Args:
        data: The images to visualize.
        show: Whether to display the image in a window.
        overlay: The images to overlay on top of the original images.
    """
    if is_bearable(data, Sequence[Image.Image]):
        images = data
    else:
        images = _convert_to_images(data)

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
def display_color_palette(n_colors: int) -> Figure:
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
    return fig


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


def plot_per_class_accuracy(
    models_dict: dict[str, nn.Module],
    dataloader: DataLoader,
    labelnames: Sequence[Any],
    img_num_per_cls: Sequence[int],
    n_classes: int = 100,
    device: torch.device = torch.device("cpu")
):
    # Adapted from https://github.com/ShadeAlsha/LTR-weight-balancing/blob/master/utils/plot_funcs.py
    result_dict = {}
    for label in models_dict:
        model = models_dict[label]
        acc_per_class = get_per_class_acc(model, dataloader, n_classes=n_classes, device=device)
        result_dict[label] = acc_per_class

    plt.figure(figsize=(15, 4), dpi=64, facecolor='w', edgecolor='k')
    plt.xticks(list(range(100)), labelnames, rotation=90, fontsize=8);  # Set text labels.
    plt.title('per-class accuracy vs. per-class #images', fontsize=20)
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    for label in result_dict:
        ax1.bar(list(range(100)), result_dict[label], alpha=0.7, width=1, label=label, edgecolor="black")

    ax1.set_ylabel('accuracy', fontsize=16, color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue', labelsize=16)

    ax2.set_ylabel('#images', fontsize=16, color='r')
    ax2.plot(img_num_per_cls, linewidth=4, color='r')
    ax2.tick_params(axis='y', labelcolor='r', labelsize=16)

    ax1.legend(prop={'size': 14})


def get_per_class_acc(model: nn.Module, dataloader: DataLoader, n_classes: int = 100, device: torch.device = torch.device("cpu")):
    predList = np.array([])
    grndList = np.array([])
    model.eval()
    for sample in dataloader:
        with torch.no_grad():
            images, labels = sample
            images = images.to(device)
            labels = labels.type(torch.long).view(-1).numpy()
            logits = model(images)
            softmaxScores = F.softmax(logits, dim=1)

            predLabels = softmaxScores.argmax(dim=1).detach().squeeze().cpu().numpy()
            predList = np.concatenate((predList, predLabels))
            grndList = np.concatenate((grndList, labels))

    confMat = metrics.confusion_matrix(grndList, predList)

    # normalize the confusion matrix
    a = confMat.sum(axis=1).reshape((-1, 1))
    confMat = confMat / a

    acc_avgClass = 0
    for i in range(confMat.shape[0]):
        acc_avgClass += confMat[i, i]

    acc_avgClass /= confMat.shape[0]

    acc_per_class = [0] * n_classes

    for i in range(n_classes):
        acc_per_class[i] = confMat[i, i]

    return acc_per_class


def plot_lr():
    lrs = []
    lr = 0.0004 / 5
    for epoch in range(60):
        if epoch < 5:
            lr += 0.0004 / 5
        elif epoch % 10 == 0:
            lr = lr / 5
        lrs.append(lr)
    plt.plot(lrs)
    plt.xlabel("Epoch")
    plt.ylabel("Learning rate")
    plt.title("Inception v3 learning rate schedule")
    plt.savefig(PROJECT_ROOT / "artifacts" / "inception3_lr_schedule.svg")
    plt.show()


def main():
    import timm
    from hw2.data.cifar100_lt import CIFAR100LT, get_img_num_per_cls

    model = timm.create_model("timm/vit_mediumd_patch16_reg4_gap_384.sbb2_e200_in12k_ft_in1k")
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    model.head = torch.nn.Linear(in_features=model.head.in_features, out_features=100)
    model.load_state_dict(torch.load(PROJECT_ROOT / "models" / "cifar100lt" / "vit_0.7399_epochs8.5.pth"))
    dataset = CIFAR100LT(root=PROJECT_ROOT / "data", train=False, download=True, transform=transforms)
    test_loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
    )
    model.to(torch.device("cuda")).eval()
    plot_per_class_accuracy({"model": model}, dataloader=test_loader, labelnames=dataset.classes,
                            img_num_per_cls=get_img_num_per_cls(100, "exp", 0.01, 500), device=torch.device("cuda"))
    plt.savefig(PROJECT_ROOT / "artifacts" / "vit_per_class_accuracy.png")


if __name__ == "__main__":
    # PCA_visualization(np.random.rand(1000, 3072), np.random.randint(0, 10, 1000), [f"Class {i}" for i in range(10)], n_components=3)
    # tsne_visualization(np.random.rand(1000, 3072), np.random.randint(0, 10, 1000), [f"Class {i}" for i in range(10)])
    main()
