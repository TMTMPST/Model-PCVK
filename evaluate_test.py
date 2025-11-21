"""
Batch Prediction & Evaluation on Test Set
Evaluasi model pada test set dan tampilkan per-class accuracy
"""

import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import json
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

print("="*80)
print("BATCH PREDICTION & EVALUATION")
print("="*80)

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load config
with open('model_config_final.json', 'r') as f:
    config = json.load(f)

num_classes = config['num_classes']
class_names = config['class_names']

print(f"Classes: {num_classes}")

# Load model
print("\nLoading model...")
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

checkpoint = torch.load('vgg16_batik_best.pth', map_location=device)
if 'model_state_dict' in checkpoint:
    vgg16.load_state_dict(checkpoint['model_state_dict'])
    best_val_acc = checkpoint.get('best_acc', 0)
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
else:
    vgg16.load_state_dict(checkpoint)

vgg16.to(device)
vgg16.eval()
print("Model loaded!")

# Transforms
test_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load test dataset
print("\nLoading test dataset...")
test_dataset = datasets.ImageFolder('data/test', transform=test_transforms)
test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)
print(f"Test samples: {len(test_dataset)}")

# Predict all
print("\nPredicting on test set...")
all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc='Testing'):
        inputs = inputs.to(device)
        outputs = vgg16(inputs)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

# Convert to numpy
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs = np.array(all_probs)

# Overall accuracy
accuracy = 100.0 * np.sum(all_preds == all_labels) / len(all_labels)

print("\n" + "="*80)
print("HASIL EVALUASI TEST SET")
print("="*80)
print(f"Total samples: {len(all_labels)}")
print(f"Overall Accuracy: {accuracy:.2f}%")
print(f"Correct predictions: {np.sum(all_preds == all_labels)}")
print(f"Wrong predictions: {np.sum(all_preds != all_labels)}")
print("="*80)

# Per-class accuracy
print("\nPER-CLASS ACCURACY:")
print("-"*80)
print(f"{'Class Name':<35} {'Samples':>10} {'Correct':>10} {'Accuracy':>12}")
print("-"*80)

class_accuracies = []
for i, class_name in enumerate(class_names):
    mask = all_labels == i
    if np.sum(mask) > 0:
        class_correct = np.sum((all_preds == all_labels) & mask)
        class_total = np.sum(mask)
        class_acc = 100.0 * class_correct / class_total
        class_accuracies.append((class_name, class_total, class_correct, class_acc))
        print(f"{class_name:<35} {class_total:>10} {class_correct:>10} {class_acc:>11.2f}%")

print("-"*80)

# Sort by accuracy
print("\nTOP 10 BEST PREDICTED CLASSES:")
sorted_by_acc = sorted(class_accuracies, key=lambda x: x[3], reverse=True)
for i, (name, total, correct, acc) in enumerate(sorted_by_acc[:10], 1):
    print(f"  {i:2d}. {name:<35} {acc:6.2f}% ({correct}/{total})")

print("\nTOP 10 WORST PREDICTED CLASSES:")
for i, (name, total, correct, acc) in enumerate(sorted_by_acc[-10:], 1):
    print(f"  {i:2d}. {name:<35} {acc:6.2f}% ({correct}/{total})")

# Find misclassified examples
print("\n" + "="*80)
print("CONTOH KESALAHAN PREDIKSI (10 pertama)")
print("="*80)
misclassified = np.where(all_preds != all_labels)[0]
print(f"Total misclassified: {len(misclassified)}")

if len(misclassified) > 0:
    print("\nSample indices yang salah diprediksi:")
    for idx in misclassified[:10]:
        true_label = class_names[all_labels[idx]]
        pred_label = class_names[all_preds[idx]]
        confidence = all_probs[idx][all_preds[idx]] * 100
        print(f"  Index {idx}: True={true_label:<30} Pred={pred_label:<30} Confidence={confidence:.2f}%")

# Confusion matrix for most confused pairs
print("\n" + "="*80)
print("MOST CONFUSED CLASS PAIRS")
print("="*80)
cm = confusion_matrix(all_labels, all_preds)
confused_pairs = []
for i in range(len(class_names)):
    for j in range(len(class_names)):
        if i != j and cm[i, j] > 0:
            confused_pairs.append((class_names[i], class_names[j], cm[i, j]))

confused_pairs.sort(key=lambda x: x[2], reverse=True)
print("Top 10 most confused pairs:")
for i, (true_class, pred_class, count) in enumerate(confused_pairs[:10], 1):
    print(f"  {i:2d}. {true_class:<30} → {pred_class:<30} ({count} kali)")

# Save detailed report
print("\n" + "="*80)
print("Saving detailed report...")
with open('test_evaluation_report.txt', 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("TEST SET EVALUATION REPORT\n")
    f.write("="*80 + "\n\n")
    
    f.write(f"Overall Accuracy: {accuracy:.2f}%\n")
    f.write(f"Total samples: {len(all_labels)}\n")
    f.write(f"Correct: {np.sum(all_preds == all_labels)}\n")
    f.write(f"Wrong: {np.sum(all_preds != all_labels)}\n\n")
    
    f.write("="*80 + "\n")
    f.write("PER-CLASS ACCURACY\n")
    f.write("="*80 + "\n")
    f.write(f"{'Class Name':<35} {'Samples':>10} {'Correct':>10} {'Accuracy':>12}\n")
    f.write("-"*80 + "\n")
    
    for name, total, correct, acc in sorted(class_accuracies, key=lambda x: x[0]):
        f.write(f"{name:<35} {total:>10} {correct:>10} {acc:>11.2f}%\n")
    
    f.write("\n" + "="*80 + "\n")
    f.write("SKLEARN CLASSIFICATION REPORT\n")
    f.write("="*80 + "\n\n")
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    f.write(report)

print("Report saved to: test_evaluation_report.txt")
print("="*80)
