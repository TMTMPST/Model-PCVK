"""
Batik Classification API Server
Using ONNX model for fast inference
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import onnxruntime as ort
import numpy as np
from PIL import Image
import cv2
import io
import json
import logging
from typing import List, Dict
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Batik Classification API",
    description="Deep learning API for Indonesian Batik pattern classification",
    version="1.0.0"
)

# Add CORS middleware for mobile app access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this in production with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and labels
ort_session = None
class_names = []
IMG_SIZE = 224

@app.on_event("startup")
async def load_model():
    """Load ONNX model and labels on server startup"""
    global ort_session, class_names
    
    try:
        # Load ONNX model
        logger.info("Loading ONNX model...")
        ort_session = ort.InferenceSession(
            "exports/vgg16_batik.onnx",
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']  # GPU if available
        )
        logger.info(f"Model loaded successfully on {ort_session.get_providers()}")
        
        # Load class labels
        with open("exports/labels.txt", "r") as f:
            class_names = [line.strip() for line in f.readlines()]
        logger.info(f"Loaded {len(class_names)} class labels")
        
        # Load model config
        with open("exports/model_config.json", "r") as f:
            config = json.load(f)
        logger.info(f"Model accuracy: {config.get('test_acc', 'N/A')}%")
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Preprocess image for model input
    Same preprocessing as training pipeline
    """
    try:
        # Load image
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image = np.array(image)
        
        # Resize to 224x224
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        
        # Bilateral Filter (noise reduction while preserving edges)
        image = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
        
        # CLAHE on LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        image = cv2.merge([l, a, b])
        image = cv2.cvtColor(image, cv2.COLOR_LAB2RGB)
        
        # Edge enhancement (unsharp masking)
        gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)
        image = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std
        
        # Convert to NCHW format (batch, channels, height, width)
        image = image.transpose(2, 0, 1)
        image = np.expand_dims(image, axis=0)
        
        return image.astype(np.float32)
    
    except Exception as e:
        logger.error(f"Preprocessing error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Image preprocessing failed: {str(e)}")

def get_predictions(probabilities: np.ndarray, top_k: int = 5) -> List[Dict]:
    """Get top-k predictions with class names and confidences"""
    top_k_idx = np.argsort(probabilities)[-top_k:][::-1]
    
    predictions = [
        {
            "class_name": class_names[idx],
            "confidence": float(probabilities[idx]),
            "confidence_percent": f"{float(probabilities[idx]) * 100:.2f}%"
        }
        for idx in top_k_idx
    ]
    
    return predictions

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Batik Classification API",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": ort_session is not None,
        "num_classes": len(class_names),
        "endpoints": {
            "/predict": "POST - Upload image for classification",
            "/health": "GET - Health check",
            "/classes": "GET - List all batik classes",
            "/docs": "GET - API documentation (Swagger UI)",
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": ort_session is not None,
        "num_classes": len(class_names),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/classes")
async def get_classes():
    """Get list of all batik classes"""
    return {
        "num_classes": len(class_names),
        "classes": class_names
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict batik class from uploaded image
    
    Parameters:
    - file: Image file (JPEG, PNG)
    
    Returns:
    - prediction: Top predicted class
    - confidence: Confidence score (0-1)
    - top_5: Top 5 predictions with confidences
    """
    
    # Validate model is loaded
    if ort_session is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image bytes
        image_bytes = await file.read()
        logger.info(f"Processing image: {file.filename} ({len(image_bytes)} bytes)")
        
        # Preprocess image
        input_tensor = preprocess_image(image_bytes)
        
        # Run inference
        ort_inputs = {ort_session.get_inputs()[0].name: input_tensor}
        ort_outputs = ort_session.run(None, ort_inputs)
        
        # Get probabilities (apply softmax)
        logits = ort_outputs[0][0]
        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / exp_logits.sum()
        
        # Get predictions
        top_5 = get_predictions(probabilities, top_k=5)
        
        # Prepare response
        response = {
            "success": True,
            "prediction": {
                "class_name": top_5[0]["class_name"],
                "confidence": top_5[0]["confidence"],
                "confidence_percent": top_5[0]["confidence_percent"]
            },
            "top_5_predictions": top_5,
            "metadata": {
                "filename": file.filename,
                "timestamp": datetime.now().isoformat(),
                "model": "VGG16 (Transfer Learning)",
                "input_size": f"{IMG_SIZE}x{IMG_SIZE}"
            }
        }
        
        logger.info(f"Prediction: {top_5[0]['class_name']} ({top_5[0]['confidence_percent']})")
        
        return JSONResponse(content=response)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict-batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Predict batik classes for multiple images (batch processing)
    
    Parameters:
    - files: List of image files
    
    Returns:
    - predictions: List of predictions for each image
    """
    
    if ort_session is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 images per batch")
    
    results = []
    
    for file in files:
        try:
            image_bytes = await file.read()
            input_tensor = preprocess_image(image_bytes)
            
            # Run inference
            ort_inputs = {ort_session.get_inputs()[0].name: input_tensor}
            ort_outputs = ort_session.run(None, ort_inputs)
            
            # Get probabilities
            logits = ort_outputs[0][0]
            exp_logits = np.exp(logits - np.max(logits))
            probabilities = exp_logits / exp_logits.sum()
            
            # Get predictions
            top_3 = get_predictions(probabilities, top_k=3)
            
            results.append({
                "filename": file.filename,
                "success": True,
                "prediction": top_3[0]
            })
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })
    
    return {
        "success": True,
        "num_images": len(files),
        "results": results
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
