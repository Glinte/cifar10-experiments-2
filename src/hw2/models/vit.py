import requests
import timm
import torch
from PIL import Image
from datasets import load_dataset
from timm.models import VisionTransformer
from torch import nn, optim
from torch.optim import lr_scheduler
from transformers import ViTImageProcessor, ViTForImageClassification

from hw2.utils import train_on_cifar


def basic_showcase():
    """Simple showcase of loading a ViT model and processing an image."""
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'

    image = Image.open(requests.get(url, stream=True).raw)
    test_dataset = load_dataset("uoft-cs/cifar10", split="test")

    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained('phuong-tk-nguyen/vit-base-patch16-224-finetuned-cifar10')

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    # model predicts one of the 1000 ImageNet classes
    predicted_class_idx = logits.argmax(-1).item()
    print("Predicted class:", model.config.id2label[predicted_class_idx])


def finetune():
    """Finetune a ViT model on CIFAR-10."""
    model: VisionTransformer = timm.create_model("timm/vit_mediumd_patch16_reg4_gap_384.sbb2_e200_in12k_ft_in1k",
                                                 pretrained=True)
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    model.head = torch.nn.Linear(in_features=model.head.in_features, out_features=10)
    # 20 blocks in total
    # Freeze all layers except the head and the last 2 blocks
    for name, param in model.named_parameters():
        # print(name)
        if "head" in name or name.startswith("blocks.19") or "fc_norm" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    epochs = 30

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters())
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer)

    train_on_cifar(model, optimizer, criterion, scheduler, transforms, epochs, device, log_run=True, cifar_dataset="10", n_test_samples=10000)
