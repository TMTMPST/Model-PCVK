"""
Quick Batik Prediction - Simple Version
Usage: python quick_predict.py path/to/image.jpg
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import sys
import os

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load config
with open('model_config_final.json', 'r') as f:
    config = json.load(f)

num_classes = config['num_classes']
class_names = config['class_names']

# Build model
vgg16 = models.vgg16(pretrained=False)
num_features = vgg16.classifier[0].in_features
vgg16.classifier = nn.Sequential(
    nn.Linear(num_features, 4096),
    nn.ReLU(inplace=True),
    nn.Dropout(0.5),
    nn.Linear(4096, 4096),
    nn.ReLU(inplace=True),
    nn.Dropout(0.5),
    nn.Linear(4096, num_classes)
)

# Load weights
checkpoint = torch.load('vgg16_batik_best.pth', map_location=device)
if 'model_state_dict' in checkpoint:
    vgg16.load_state_dict(checkpoint['model_state_dict'])
else:
    vgg16.load_state_dict(checkpoint)

vgg16.to(device)
vgg16.eval()

# Transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Predict
if len(sys.argv) < 2:
    print("Usage: python quick_predict.py path/to/image.jpg")
    sys.exit(1)

image_path = sys.argv[1]

if not os.path.exists(image_path):
    print(f"Error: File tidak ditemukan: {image_path}")
    sys.exit(1)

# Load image
image = Image.open(image_path).convert('RGB')
input_tensor = transform(image).unsqueeze(0).to(device)

# Inference
with torch.no_grad():
    outputs = vgg16(input_tensor)
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    confidence, predicted = torch.max(probabilities, 1)

# Top 5
topk_prob, topk_idx = torch.topk(probabilities, min(5, len(class_names)))

# Display results
print("\n" + "="*80)
print("HASIL PREDIKSI BATIK")
print("="*80)
print(f"File: {os.path.basename(image_path)}")
print(f"Size: {image.size[0]}x{image.size[1]} pixels")
print()
print(f"Motif Prediksi: {class_names[predicted.item()]}")
print(f"Confidence: {confidence.item() * 100:.2f}%")
print()
print("Top 5 Predictions:")
for i, (idx, prob) in enumerate(zip(topk_idx[0], topk_prob[0]), 1):
    bar = "█" * int(prob.item() * 50)
    print(f"  {i}. {class_names[idx]:35s} {prob.item() * 100:6.2f}% {bar}")
print("="*80)
