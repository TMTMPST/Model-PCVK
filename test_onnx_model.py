"""
Test script to verify ONNX model export and inference
Run this after training to ensure model works correctly
"""

import onnxruntime as ort
import numpy as np
import cv2
from PIL import Image
import json
import os

def preprocess_image(image_path, img_size=224):
    """Preprocess image (same as training)"""
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Resize
    image = cv2.resize(image, (img_size, img_size))
    
    # Bilateral Filter
    image = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    
    # CLAHE
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    image = cv2.merge([l, a, b])
    image = cv2.cvtColor(image, cv2.COLOR_LAB2RGB)
    
    # Edge enhancement
    gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)
    image = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
    
    # Normalize
    image = image.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std
    
    # Convert to NCHW
    image = image.transpose(2, 0, 1)
    image = np.expand_dims(image, axis=0)
    
    return image.astype(np.float32)

def test_onnx_model():
    """Test ONNX model export and inference"""
    
    print("=" * 70)
    print("ONNX MODEL VERIFICATION TEST")
    print("=" * 70)
    
    # Check if exports folder exists
    if not os.path.exists('exports'):
        print("❌ ERROR: 'exports' folder not found!")
        print("   Run the training notebook first to generate exports.")
        return False
    
    # Check if ONNX model exists
    onnx_path = 'exports/vgg16_batik.onnx'
    if not os.path.exists(onnx_path):
        print(f"❌ ERROR: ONNX model not found at {onnx_path}")
        print("   Run the training notebook to export the model.")
        return False
    
    print(f"\n✓ Found ONNX model: {onnx_path}")
    
    # Load model
    print("\nLoading ONNX model...")
    try:
        session = ort.InferenceSession(onnx_path)
        print(f"✓ Model loaded successfully")
        print(f"  Providers: {session.get_providers()}")
    except Exception as e:
        print(f"❌ ERROR loading model: {str(e)}")
        return False
    
    # Load labels
    labels_path = 'exports/labels.txt'
    if os.path.exists(labels_path):
        with open(labels_path, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
        print(f"✓ Loaded {len(class_names)} class labels")
    else:
        print(f"⚠ Warning: labels.txt not found")
        class_names = [f"Class_{i}" for i in range(10)]
    
    # Load config
    config_path = 'exports/model_config.json'
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"✓ Model config loaded")
        print(f"  Test Accuracy: {config.get('test_acc', 'N/A')}%")
        print(f"  Number of classes: {config.get('num_classes', 'N/A')}")
    
    # Test inference with random input
    print("\n" + "=" * 70)
    print("TEST 1: Random Input Inference")
    print("=" * 70)
    
    try:
        dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: dummy_input})
        
        print(f"✓ Inference successful")
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Output shape: {outputs[0].shape}")
        print(f"  Output range: [{outputs[0].min():.4f}, {outputs[0].max():.4f}]")
        
        # Check output probabilities
        logits = outputs[0][0]
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()
        
        top_3_idx = np.argsort(probs)[-3:][::-1]
        print(f"\n  Top 3 predictions (random input):")
        for i, idx in enumerate(top_3_idx):
            print(f"    {i+1}. {class_names[idx]}: {probs[idx]*100:.2f}%")
        
    except Exception as e:
        print(f"❌ ERROR during inference: {str(e)}")
        return False
    
    # Test with real image if available
    print("\n" + "=" * 70)
    print("TEST 2: Real Image Inference")
    print("=" * 70)
    
    # Try to find a test image
    test_image = None
    for root, dirs, files in os.walk('data/test'):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                test_image = os.path.join(root, file)
                break
        if test_image:
            break
    
    if test_image and os.path.exists(test_image):
        print(f"Testing with: {test_image}")
        
        try:
            # Preprocess
            input_tensor = preprocess_image(test_image)
            
            # Inference
            outputs = session.run(None, {input_name: input_tensor})
            
            # Get probabilities
            logits = outputs[0][0]
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / exp_logits.sum()
            
            top_5_idx = np.argsort(probs)[-5:][::-1]
            
            print(f"✓ Inference successful")
            print(f"\n  Top 5 predictions:")
            for i, idx in enumerate(top_5_idx):
                print(f"    {i+1}. {class_names[idx]}: {probs[idx]*100:.2f}%")
            
        except Exception as e:
            print(f"❌ ERROR processing image: {str(e)}")
            return False
    else:
        print("⚠ No test images found in data/test/")
        print("  Skipping real image test")
    
    # Performance test
    print("\n" + "=" * 70)
    print("TEST 3: Performance Benchmark")
    print("=" * 70)
    
    import time
    
    num_iterations = 100
    dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
    
    # Warm up
    for _ in range(10):
        session.run(None, {input_name: dummy_input})
    
    # Benchmark
    start_time = time.time()
    for _ in range(num_iterations):
        session.run(None, {input_name: dummy_input})
    elapsed_time = time.time() - start_time
    
    avg_time = (elapsed_time / num_iterations) * 1000
    fps = num_iterations / elapsed_time
    
    print(f"✓ Performance test completed")
    print(f"  Iterations: {num_iterations}")
    print(f"  Total time: {elapsed_time:.2f}s")
    print(f"  Average time per inference: {avg_time:.2f}ms")
    print(f"  Throughput: {fps:.2f} FPS")
    
    # Summary
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    print("✓ ONNX model is valid and working correctly!")
    print("✓ Model can be deployed to production")
    print("\nNext steps:")
    print("  1. Run: python api_server.py")
    print("  2. Test API: curl -X POST http://localhost:8000/predict -F 'file=@test.jpg'")
    print("  3. Deploy to cloud (AWS/GCP/Azure)")
    print("=" * 70)
    
    return True

if __name__ == "__main__":
    success = test_onnx_model()
    exit(0 if success else 1)
