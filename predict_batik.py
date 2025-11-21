"""
VGG16 Batik Classification - Inference Script
Gunakan script ini untuk menebak motif batik dari gambar
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import os
import sys

def load_model(model_path, config_path, device):
    """Load trained model"""
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    num_classes = config['num_classes']
    class_names = config['class_names']
    
    # Build model architecture
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
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        vgg16.load_state_dict(checkpoint['model_state_dict'])
    else:
        vgg16.load_state_dict(checkpoint)
    
    vgg16.to(device)
    vgg16.eval()
    
    return vgg16, class_names


def get_transforms():
    """Get image preprocessing transforms"""
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])


def predict_image(image_path, model, class_names, transform, device, top_k=5):
    """Predict batik motif from image"""
    # Load and preprocess image
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image: {e}")
        return None
    
    # Show image info
    print(f"\nImage: {os.path.basename(image_path)}")
    print(f"Size: {image.size[0]}x{image.size[1]} pixels")
    
    # Preprocess
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    # Get top-k predictions
    topk_prob, topk_idx = torch.topk(probabilities, min(top_k, len(class_names)))
    
    # Results
    predicted_class = class_names[predicted.item()]
    confidence_score = confidence.item() * 100
    
    top_predictions = [
        (class_names[idx], prob.item() * 100)
        for idx, prob in zip(topk_idx[0], topk_prob[0])
    ]
    
    return predicted_class, confidence_score, top_predictions


def main():
    print("="*80)
    print("VGG16 BATIK CLASSIFICATION - INFERENCE")
    print("="*80)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # Model paths
    model_path = 'vgg16_batik_best.pth'
    config_path = 'model_config_final.json'
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found: {model_path}")
        print("Please train the model first!")
        return
    
    if not os.path.exists(config_path):
        print(f"ERROR: Config file not found: {config_path}")
        print("Please train the model first!")
        return
    
    # Load model
    print("Loading model...")
    model, class_names = load_model(model_path, config_path, device)
    transform = get_transforms()
    print(f"Model loaded! ({len(class_names)} classes)")
    print("="*80)
    
    # Interactive mode
    while True:
        print("\nOptions:")
        print("  1. Predict single image")
        print("  2. Predict multiple images")
        print("  3. Exit")
        
        choice = input("\nPilih (1/2/3): ").strip()
        
        if choice == '1':
            # Single image prediction
            image_path = input("\nMasukkan path gambar: ").strip().strip('"').strip("'")
            
            if not os.path.exists(image_path):
                print(f"ERROR: File tidak ditemukan: {image_path}")
                continue
            
            result = predict_image(image_path, model, class_names, transform, device)
            
            if result:
                predicted_class, confidence, top_predictions = result
                
                print("\n" + "="*80)
                print("HASIL PREDIKSI")
                print("="*80)
                print(f"Motif: {predicted_class}")
                print(f"Confidence: {confidence:.2f}%")
                print(f"\nTop 5 Predictions:")
                for i, (cls, prob) in enumerate(top_predictions, 1):
                    bar = "█" * int(prob / 2)
                    print(f"  {i}. {cls:35s} {prob:6.2f}% {bar}")
                print("="*80)
        
        elif choice == '2':
            # Multiple images prediction
            folder_path = input("\nMasukkan path folder: ").strip().strip('"').strip("'")
            
            if not os.path.exists(folder_path):
                print(f"ERROR: Folder tidak ditemukan: {folder_path}")
                continue
            
            # Get all image files
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
            image_files = [
                os.path.join(folder_path, f) 
                for f in os.listdir(folder_path)
                if os.path.splitext(f.lower())[1] in image_extensions
            ]
            
            if not image_files:
                print("Tidak ada gambar ditemukan di folder tersebut!")
                continue
            
            print(f"\nDitemukan {len(image_files)} gambar. Memproses...\n")
            
            results = []
            for image_path in image_files:
                result = predict_image(image_path, model, class_names, transform, device)
                if result:
                    predicted_class, confidence, _ = result
                    results.append({
                        'file': os.path.basename(image_path),
                        'motif': predicted_class,
                        'confidence': confidence
                    })
                    print(f"✓ {os.path.basename(image_path):30s} → {predicted_class:30s} ({confidence:.1f}%)")
            
            # Summary
            print("\n" + "="*80)
            print(f"SELESAI - Total: {len(results)} gambar")
            print("="*80)
            
            # Save results
            save = input("\nSimpan hasil ke file? (y/n): ").strip().lower()
            if save == 'y':
                output_file = 'prediction_results.txt'
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write("HASIL PREDIKSI BATIK\n")
                    f.write("="*80 + "\n\n")
                    for r in results:
                        f.write(f"File: {r['file']}\n")
                        f.write(f"Motif: {r['motif']}\n")
                        f.write(f"Confidence: {r['confidence']:.2f}%\n")
                        f.write("-"*80 + "\n")
                print(f"Hasil disimpan ke: {output_file}")
        
        elif choice == '3':
            print("\nTerima kasih!")
            break
        
        else:
            print("Pilihan tidak valid!")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram dihentikan.")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
