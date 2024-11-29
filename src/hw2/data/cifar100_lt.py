import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, List, Tuple, Literal

import numpy as np

from hw2.visualization import visualize_images
from matplotlib import patches, pyplot as plt
from PIL import Image
import torch
from matplotlib.figure import Figure
from torchvision.datasets import CIFAR100, VisionDataset
from torchvision.transforms import v2
from typing_extensions import override

from hw2 import PROJECT_ROOT, CIFAR100LT_NORMALIZATION


class CIFAR100LT(CIFAR100):
    """CIFAR-100 Long-Tailed (Imbalanced) Dataset.

    This dataset creates an imbalanced version of the CIFAR-100 dataset,
    where the number of images per class follows a specified distribution
    (e.g., exponential or step function).

    Attributes:
        root (str or Path): Root directory of the dataset.
        train (bool): If True, creates dataset from the training set, otherwise from the test set.
        imb_type (str): Type of imbalance ('exp' or 'step').
        imb_factor (float): Imbalance factor controlling the degree of imbalance.
        transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        data (np.ndarray): The data array containing all images.
        targets (list[int]): The list of target labels corresponding to the data.
        num_classes (int): The total number of classes in the dataset.
        classes (list[str]): The list of class names.
        class_to_idx (dict): A mapping from class names to class indices.
        superclass_to_class (dict): A mapping from superclass names to a list of subclass names.
        meta (dict): Metadata of the dataset.
        basic_train_transform (callable): A good default transform for training images. Not used by default.
        basic_test_transform (callable): A good default transform for test images. Not used by default.
    """

    basic_train_transform = v2.Compose(
        [
            v2.ToImage(),
            v2.RandomHorizontalFlip(),
            v2.RandomAffine(degrees=10, translate=(0.1, 0.1)),
            v2.RandomCrop(32, padding=4),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(*CIFAR100LT_NORMALIZATION),
            # v2.ToPILImage(),  # Uncomment this line to visualize the transformed images
        ]
    )

    basic_test_transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(*CIFAR100LT_NORMALIZATION),
        ]
    )

    superclass_to_class = {
        "aquatic mammals": ["beaver", "dolphin", "otter", "seal", "whale"],
        "fish": ["aquarium_fish", "flatfish", "ray", "shark", "trout"],
        "flowers": ["orchid", "poppy", "rose", "sunflower", "tulip"],
        "food containers": ["bottle", "bowl", "can", "cup", "plate"],
        "fruit and vegetables": ["apple", "mushroom", "orange", "pear", "sweet_pepper"],
        "household electrical devices": ["clock", "keyboard", "lamp", "telephone", "television"],
        "household furniture": ["bed", "chair", "couch", "table", "wardrobe"],
        "insects": ["bee", "beetle", "butterfly", "caterpillar", "cockroach"],
        "large carnivores": ["bear", "leopard", "lion", "tiger", "wolf"],
        "large man-made outdoor things": ["bridge", "castle", "house", "road", "skyscraper"],
        "large natural outdoor scenes": ["cloud", "forest", "mountain", "plain", "sea"],
        "large omnivores and herbivores": ["camel", "cattle", "chimpanzee", "elephant", "kangaroo"],
        "medium-sized mammals": ["fox", "porcupine", "possum", "raccoon", "skunk"],
        "non-insect invertebrates": ["crab", "lobster", "snail", "spider", "worm"],
        "people": ["baby", "boy", "girl", "man", "woman"],
        "reptiles": ["crocodile", "dinosaur", "lizard", "snake", "turtle"],
        "small mammals": ["hamster", "mouse", "rabbit", "shrew", "squirrel"],
        "trees": ["maple_tree", "oak_tree", "palm_tree", "pine_tree", "willow_tree"],
        "vehicles 1": ["bicycle", "bus", "motorcycle", "pickup_truck", "train"],
        "vehicles 2": ["lawn_mower", "rocket", "streetcar", "tank", "tractor"],
    }

    def __init__(
        self,
        root: str | Path | None = None,
        train: bool = True,
        imb_type: Literal["exp", "step"] = 'exp',
        imb_factor: float = 0.01,
        transform: Callable[[Image.Image], Any] | None = None,
        target_transform: Callable[[int], Any] | None = None,
        download: bool = False,
    ) -> None:
        """
        Initialize the CIFAR-100 Long-Tailed (Imbalanced) Dataset.

        Args:
            root (str or Path): Root directory of the dataset.
            train (bool, optional): If True, creates dataset from the training set, otherwise from the test set.
            imb_type (str, optional): Type of imbalance ('exp' or 'step'). Default is 'exp'.
            imb_factor (float, optional): Imbalance factor controlling the degree of imbalance. Should be between 0 and 1. Default is 0.01.
            transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed version.
            target_transform (callable, optional): A function/transform that takes in the target and transforms it.
            download (bool, optional): If True, downloads the dataset if it doesn't exist in the root directory.
        """
        if root is None:
            root = "."  # Use current directory if root is not specified
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

        self.train = train
        self.imb_type = imb_type
        self.imb_factor = imb_factor
        self.num_classes = 100  # CIFAR-100 has 100 classes

        if self.train:
            # Generate the imbalanced dataset
            self.img_num_per_cls = self._get_img_num_per_cls()
            self.data, self.targets = self._gen_imbalanced_data(self.data, self.targets, self.img_num_per_cls)

    def _get_img_num_per_cls(self) -> List[int]:
        """
        This method calculates the desired number of samples for each class based on the specified imbalance type (`imb_type`) and imbalance factor (`imb_factor`). It simulates different imbalanced data distributions to reflect real-world scenarios where class frequencies are not uniform.

        Returns:
            list[int]: A list containing the computed number of images for each class after applying the specified imbalance. The length of the list is equal to `self.num_classes`, and each element corresponds to a class index.
        """

        return get_img_num_per_cls(self.num_classes, self.imb_type, self.imb_factor, img_max=len(self.data) // self.num_classes)

    @staticmethod
    def _gen_imbalanced_data(
        data: np.ndarray,
        targets: list[int],
        img_num_per_cls: list[int]
    ) -> tuple[np.ndarray, list[int]]:
        """
        Generate an imbalanced dataset by selecting a subset of data and targets according to the specified number of images per class.

        Args:
            data (np.ndarray): The original data array containing all images.
            targets (list[int]): The list of original target labels corresponding to the data.
            img_num_per_cls (list[int]): A list containing the desired number of images per class.

        Returns:
            tuple[np.ndarray, list[int]]: A tuple containing:
                - new_data (np.ndarray): The data array after selecting the specified number of samples per class.
                - new_targets (List[int]): The list of target labels corresponding to `new_data`.
        """
        # This function is adapted from publicly available code [commit 6feb304, MIT License]:
        # https://github.com/kaidic/LDAM-DRW/blob/master/imbalance_cifar.py

        new_data = []
        new_targets = []

        targets_np = np.array(targets)
        classes = np.unique(targets_np)

        if len(img_num_per_cls) != len(classes):
            raise ValueError("Number of classes and number of samples per class do not match.")

        for cls_idx, cls_img_num in zip(classes, img_num_per_cls):
            cls_indices = np.where(targets_np == cls_idx)[0]
            np.random.shuffle(cls_indices)
            selected_indices = cls_indices[:cls_img_num]
            new_data.append(data[selected_indices])
            new_targets.extend([cls_idx] * cls_img_num)

        new_data = np.vstack(new_data)
        return new_data, new_targets

    def get_superclass_sample(self):
        """
        Takes a CIFAR-100 dataset and returns a dictionary of superclass samples.

        Returns:
            dict: {superclass: [5 images (one from each subclass)]}
        """
        superclass_samples = {}

        for superclass, subclasses in self.superclass_to_class.items():
            subclass_indices = [self.class_to_idx[subclass] for subclass in subclasses]

            images = []
            for subclass_idx in subclass_indices:
                subclass_samples = [i for i, label in enumerate(self.targets) if label == subclass_idx]
                random_sample_idx = random.choice(subclass_samples)
                images.append(self[random_sample_idx][0])  # Add the image (not label) to the list

            superclass_samples[superclass] = images

        return superclass_samples

    def visualize(self) -> Figure:
        fig, axes = plt.subplots(10, 10, figsize=(18, 18))
        for i, (cls, superclass_samples) in enumerate(self.get_superclass_sample().items()):
            for j, img in enumerate(superclass_samples):
                axes[i // 2, j + 5 * (i % 2)].imshow(img)
                axes[i // 2, j + 5 * (i % 2)].axis("off")
                axes[i // 2, j + 5 * (i % 2)].set_title(self.superclass_to_class[cls][j])

            # Format the superclass name to split words into separate lines, because the graph has overlapping text without this
            formatted_superclass = "\n".join(cls.split())

            # Add superclass labels
            if i % 2 == 0:  # Left side
                axes[i // 2, 0].annotate(
                    formatted_superclass,
                    xy=(-0.4, 0.5),
                    xycoords="axes fraction",
                    fontsize=10,
                    ha="center",
                    va="center",
                    rotation=90,
                    wrap=True,
                )
            else:  # Right side
                axes[i // 2, 9].annotate(
                    formatted_superclass,
                    xy=(1.4, 0.5),
                    xycoords="axes fraction",
                    fontsize=10,
                    ha="center",
                    va="center",
                    rotation=270,
                    wrap=True,
                )

        for ax_row in axes:
            ax_row[4].add_patch(
                patches.Rectangle(
                    xy=(1.08, -0.05),
                    width=0.02,
                    height=1.4,
                    transform=ax_row[4].transAxes,
                    color="black",
                    clip_on=False,
                )
            )

        return fig

def get_img_num_per_cls(cls_num: int, imb_type: Literal["exp", "step"], imb_factor: float, img_max: int) -> List[int]:
    """
    This method calculates the desired number of samples for each class based on the specified imbalance type (`imb_type`) and imbalance factor (`imb_factor`). It simulates different imbalanced data distributions to reflect real-world scenarios where class frequencies are not uniform.

    Args:
        cls_num (int): The total number of classes in the dataset.
        imb_type (str): The type of imbalance to introduce. Supported options are:
            - 'exp': Exponential decay in the number of samples per class.
              The number of samples decreases exponentially from the first class to the last, controlled by `imb_factor`.
            - 'step': Step imbalance where the first half of the classes have the maximum number of samples, and the second half have fewer samples. The reduction in samples for the minority classes is determined by `imb_factor`.
        imb_factor (float): The imbalance factor controlling the degree of imbalance between classes.
            - For 'exp' type: Determines the rate of exponential decay. A smaller value leads to a steeper decline in class samples.
            - For 'step' type: Represents the ratio between the number of samples in minority classes and majority classes.
            - The value should be between 0 and 1, where values closer to 0 increase the level of imbalance.
        img_max (int): Maximum number of images per class when balanced.

    Returns:
        list[int]: A list containing the computed number of images for each class after applying the specified imbalance. The length of the list is equal to `cls_num`, and each element corresponds to a class index.
    """
    # This function is adapted from publicly available code [commit 6feb304, MIT License]:
    # https://github.com/kaidic/LDAM-DRW/blob/master/imbalance_cifar.py

    if imb_factor < 0 or imb_factor > 1:
        raise ValueError("Imbalance factor should be between 0 and 1.")

    img_num_per_cls = []

    if imb_type == 'exp':
        # Exponential decay of class samples
        for cls_idx in range(cls_num):
            num = img_max * (imb_factor ** (cls_idx / (cls_num - 1)))
            img_num_per_cls.append(int(np.round(num)))
    elif imb_type == 'step':
        # Step imbalance: half classes have img_max samples, rest have img_max * imb_factor
        for cls_idx in range(cls_num):
            if cls_idx < cls_num / 2:
                img_num_per_cls.append(img_max)
            else:
                img_num_per_cls.append(int(img_max * imb_factor))
    else:
        raise ValueError(f"Invalid imbalance type: {imb_type}. Supported types are 'exp' and 'step'.")

    return img_num_per_cls


def main():
    dataset = CIFAR100LT(root=PROJECT_ROOT / "data", train=True, imb_type='exp', imb_factor=0.01, download=True, transform=CIFAR100LT.basic_train_transform)
    # print(dataset.get_superclass_sample())
    # print(CIFAR100LT_NORMALIZATION)
    fig = dataset.visualize()
    plt.suptitle("CIFAR-100 Long-Tailed (Imbalanced) Dataset, after transformations", fontsize=24)
    fig.savefig(PROJECT_ROOT / "artifacts" / "CISC3027 hw3" / "cifar100lt_samples_trasnformed.svg")
    fig.show()


if __name__ == "__main__":
    main()
