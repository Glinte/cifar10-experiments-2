import requests
from PIL import Image
from datasets import load_dataset
from transformers import ViTImageProcessor, ViTForImageClassification

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