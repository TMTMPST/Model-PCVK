"""
Batik Classification Web App - Gradio (20 Classes - Classical Features)
Upload gambar batik dan model akan mendeteksi motifnya menggunakan HSV+GLCM+LBP+SVM!
Model dengan 20 motif pilihan - Akurasi 97.29%
"""

import gradio as gr
import numpy as np
import cv2
from PIL import Image
import onnxruntime as ort
import json
import os
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from scipy.stats import skew
import warnings
warnings.filterwarnings('ignore')

# Setup
print("Loading ONNX model (20 Classes)...")

# Load the trained ONNX model
MODEL_PATH = "Batik_Group7(20C).onnx"
CONFIG_PATH = "batik_config(20C).json"

# Initialize default config
config = {
    'class_names': [],
    'IMG_SIZE': [256, 256],
    'LBP_RADIUS': 3,
    'LBP_N_POINTS': 24,
    'GLCM_DISTANCES': [1],
    'GLCM_ANGLES': [0, 0.785398, 1.570796, 2.356194]
}
class_names = []

# Load configuration
if os.path.exists(CONFIG_PATH):
    try:
        with open(CONFIG_PATH, 'r') as f:
            loaded_config = json.load(f)
        config.update(loaded_config)  # Update default config with loaded values
        class_names = config.get('class_names', [])
        print(f"✓ Configuration loaded from {CONFIG_PATH}")
        print(f"✓ Number of classes: {len(class_names)}")
    except Exception as e:
        print(f"[WARNING] Could not load config: {e}")
else:
    print(f"[WARNING] Config file not found: {CONFIG_PATH}")
    print(f"  Using default configuration")

# Load ONNX model
try:
    onnx_session = ort.InferenceSession(MODEL_PATH)
    print(f"✓ ONNX model loaded: {MODEL_PATH}")

    # Get model input info
    input_name = onnx_session.get_inputs()[0].name
    input_shape = onnx_session.get_inputs()[0].shape
    print(f"✓ Model input: {input_name}, shape: {input_shape}")

    # Get model output info
    output_names = [output.name for output in onnx_session.get_outputs()]
    print(f"✓ Model outputs: {output_names}")

except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    onnx_session = None

# Feature extraction functions
def extract_hsv_features(image):
    """Extract HSV color moments (mean, std, skewness) for each channel"""
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    features = []

    for i in range(3):
        channel = hsv[:, :, i].flatten()
        features.extend([
            np.mean(channel),
            np.std(channel),
            skew(channel)
        ])

    return np.array(features)

def extract_glcm_features(image, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    """Extract GLCM texture features"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Normalize to 0-255 range
    gray = ((gray - gray.min()) / (gray.max() - gray.min() + 1e-7) * 255).astype(np.uint8)

    # Compute GLCM
    glcm = graycomatrix(
        gray,
        distances=distances,
        angles=angles,
        levels=256,
        symmetric=True,
        normed=True
    )

    # Extract properties
    properties = ['contrast', 'dissimilarity', 'homogeneity',
                  'energy', 'correlation', 'ASM']

    features = []
    for prop in properties:
        values = graycoprops(glcm, prop).flatten()
        features.append(np.mean(values))

    return np.array(features)

def extract_lbp_features(image, radius=3, n_points=24, n_bins=26):
    """Extract LBP (Local Binary Pattern) features"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Compute LBP
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')

    # Compute histogram
    hist, _ = np.histogram(
        lbp.ravel(),
        bins=n_bins,
        range=(0, n_bins),
        density=True
    )

    return hist

def extract_combined_features(image):
    """Extract all features: HSV + GLCM + LBP"""
    try:
        # Resize image
        img_size = tuple(config['IMG_SIZE'])
        image_resized = cv2.resize(image, img_size)

        # Extract features
        hsv_features = extract_hsv_features(image_resized)
        glcm_features = extract_glcm_features(
            image_resized,
            distances=config['GLCM_DISTANCES'],
            angles=config['GLCM_ANGLES']
        )
        lbp_features = extract_lbp_features(
            image_resized,
            radius=config['LBP_RADIUS'],
            n_points=config['LBP_N_POINTS']
        )

        # Combine all features
        combined = np.concatenate([hsv_features, glcm_features, lbp_features])

        return combined

    except Exception as e:
        print(f"Error in feature extraction: {e}")
        return None

# Prediction function
def predict_batik(image):
    """Predict batik motif from uploaded image"""
    try:
        if image is None:
            return "❌ Tidak ada gambar yang di-upload!", None

        # Convert PIL to CV2 format
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Convert RGB to BGR for OpenCV
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image

        # Extract features
        features = extract_combined_features(image_bgr)

        if features is None:
            return "❌ Gagal mengekstrak fitur dari gambar!", None

        # Prepare input for ONNX model
        features_input = features.reshape(1, -1).astype(np.float32)

        # Run inference
        outputs = onnx_session.run(None, {input_name: features_input})

        # Parse outputs - handle different ONNX output formats
        if len(outputs) >= 2:
            # Format 1: First output is label, second is probabilities
            if isinstance(outputs[0], np.ndarray) and outputs[0].dtype.kind in ('U', 'S', 'O'):
                # String labels directly
                predicted_label = str(outputs[0][0])
                probabilities = outputs[1][0] if len(outputs) > 1 else None
            elif isinstance(outputs[1], np.ndarray) and len(outputs[1].shape) > 0:
                # Probabilities in second output
                probabilities = outputs[1][0] if len(outputs[1].shape) == 2 else outputs[1]
                predicted_idx = np.argmax(probabilities)
                predicted_label = class_names[predicted_idx] if predicted_idx < len(class_names) else f"Class_{predicted_idx}"
            else:
                # Integer index in first output
                predicted_idx = int(outputs[0][0])
                predicted_label = class_names[predicted_idx] if predicted_idx < len(class_names) else f"Class_{predicted_idx}"
                probabilities = outputs[1][0] if len(outputs) > 1 else None
        else:
            # Single output - could be probabilities or label
            output = outputs[0]
            if output.dtype.kind in ('U', 'S', 'O'):
                # String label
                predicted_label = str(output[0])
                probabilities = None
            elif len(output.shape) == 2 and output.shape[1] > 1:
                # Probabilities array
                probabilities = output[0]
                predicted_idx = np.argmax(probabilities)
                predicted_label = class_names[predicted_idx] if predicted_idx < len(class_names) else f"Class_{predicted_idx}"
            else:
                # Single integer index
                predicted_idx = int(output[0])
                predicted_label = class_names[predicted_idx] if predicted_idx < len(class_names) else f"Class_{predicted_idx}"
                probabilities = None

        # Get confidence score
        if probabilities is not None and len(probabilities) > 0:
            confidence = float(np.max(probabilities)) * 100
        else:
            confidence = 100.0  # If no probabilities, assume 100% for the predicted class

        # Format prediction result
        region = predicted_label.split('_')[0] if '_' in predicted_label else "Unknown"
        motif = predicted_label.split('_')[1] if '_' in predicted_label else predicted_label

        result_html = f"""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    padding: 30px; border-radius: 15px; color: white; text-align: center;
                    box-shadow: 0 10px 25px rgba(0,0,0,0.2);'>
            <h2 style='margin: 0 0 20px 0; font-size: 2em;'>🎨 Hasil Prediksi</h2>
            <div style='background: rgba(255,255,255,0.15); padding: 20px; border-radius: 10px; margin: 15px 0;
                        backdrop-filter: blur(10px);'>
                <p style='margin: 5px 0; font-size: 1.1em;'><strong>Motif:</strong> {motif}</p>
                <p style='margin: 5px 0; font-size: 1.1em;'><strong>Asal:</strong> {region}</p>
                <p style='margin: 5px 0; font-size: 1.1em;'><strong>Label:</strong> {predicted_label}</p>
            </div>
            <div style='background: rgba(255,255,255,0.2); padding: 15px; border-radius: 10px;
                        backdrop-filter: blur(10px);'>
                <h3 style='margin: 0 0 10px 0;'>📊 Confidence Score</h3>
                <p style='font-size: 2.5em; margin: 0; font-weight: bold;'>{confidence:.2f}%</p>
            </div>
        </div>
        """

        # Get top 10 predictions if we have probabilities
        top_10_html = ""
        if probabilities is not None and len(class_names) > 0:
            top_10_html = "<div style='margin-top: 20px; padding: 20px; background: #f8f9fa; border-radius: 10px;'>"
            top_10_html += "<h3 style='color: #333; margin-bottom: 15px;'>🔝 Top 10 Prediksi Teratas</h3>"

            # Get top 10 indices
            top_10_indices = np.argsort(probabilities)[-10:][::-1]

            for idx in top_10_indices:
                if idx < len(class_names):
                    class_label = class_names[idx]
                    prob = probabilities[idx] * 100

                    # Create progress bar
                    bar_width = int(prob)
                    color = '#667eea' if idx == top_10_indices[0] else '#8e9eef'

                    top_10_html += f"""
                    <div style='margin: 10px 0;'>
                        <div style='display: flex; justify-content: space-between; margin-bottom: 5px;'>
                            <span style='color: #333; font-weight: 500;'>{class_label}</span>
                            <span style='color: #666;'>{prob:.2f}%</span>
                        </div>
                        <div style='background: #e0e0e0; border-radius: 5px; overflow: hidden; height: 8px;'>
                            <div style='background: {color}; width: {bar_width}%; height: 100%; transition: width 0.3s;'></div>
                        </div>
                    </div>
                    """

            top_10_html += "</div>"

        return result_html + top_10_html, None

    except Exception as e:
        error_html = f"""
        <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                    padding: 30px; border-radius: 15px; color: white; text-align: center;
                    box-shadow: 0 10px 25px rgba(0,0,0,0.2);'>
            <h2 style='margin: 0 0 20px 0;'>❌ Error</h2>
            <p style='font-size: 1.2em;'>Terjadi kesalahan saat prediksi:</p>
            <p style='background: rgba(255,255,255,0.2); padding: 15px; border-radius: 10px;
                      font-family: monospace; margin-top: 15px;'>{str(e)}</p>
        </div>
        """
        return error_html, None

# Custom CSS for better UI
custom_css = """
.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
}
.contain {
    max-height: 600px !important;
}
footer {
    display: none !important;
}
"""

# Create Gradio interface
print("Creating Gradio interface...")

with gr.Blocks() as demo:
    # Inject CSS
    gr.HTML(f"<style>{custom_css}</style>")

    gr.HTML("""
        <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    border-radius: 15px; margin-bottom: 20px; color: white;'>
            <h1 style='margin: 0; font-size: 2.5em;'>🎨 Batik Classification (20 Motif)</h1>
            <p style='margin: 10px 0 0 0; font-size: 1.2em; opacity: 0.9;'>
                Upload gambar batik dan AI akan mengenali motifnya! 🤖<br/>
                <span style='font-size: 0.9em;'>Model: HSV + GLCM + LBP + SVM | Akurasi: 97.29% | 20 Motif Pilihan</span>
            </p>
        </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(
                label="📸 Upload Gambar Batik",
                type="pil",
                height=400
            )

            predict_btn = gr.Button(
                "🔮 Prediksi Motif!",
                variant="primary",
                size="lg"
            )

            gr.HTML("""
                <div style='margin-top: 20px; padding: 15px; background: #e7f3ff; border-radius: 10px; border-left: 4px solid #2196F3;'>
                    <h4 style='margin: 0 0 10px 0; color: #1976D2;'>💡 Tips:</h4>
                    <ul style='margin: 0; padding-left: 20px; color: #333;'>
                        <li>Upload gambar dengan kualitas bagus</li>
                        <li>Pastikan motif batik terlihat jelas</li>
                        <li>Format: JPG, PNG, JPEG</li>
                        <li>Model dilatih dengan 20 motif pilihan</li>
                    </ul>
                </div>
            """)

        with gr.Column(scale=1):
            output_html = gr.HTML(label="🎯 Hasil Prediksi")

            # Available classes info
            if class_names:
                classes_html = "<div style='margin-top: 20px; padding: 15px; background: #f5f5f5; border-radius: 10px;'>"
                classes_html += f"<h4 style='color: #333; margin-bottom: 10px;'>📋 20 Motif yang Tersedia ({len(class_names)} kelas):</h4>"
                classes_html += "<div style='display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px;'>"

                for class_name in sorted(class_names):
                    classes_html += f"<div style='background: white; padding: 8px; border-radius: 5px; font-size: 0.9em; color: #555;'>• {class_name}</div>"

                classes_html += "</div></div>"
                gr.HTML(classes_html)

    # Connect button to prediction function
    predict_btn.click(
        fn=predict_batik,
        inputs=input_image,
        outputs=[output_html, gr.State()]
    )

    gr.HTML("""
        <div style='text-align: center; padding: 15px; margin-top: 20px; color: #666; border-top: 1px solid #ddd;'>
            <p style='margin: 0;'>🎓 Batik Classification System | PBL Kelompok 7 | Politeknik Caltex Riau</p>
            <p style='margin: 5px 0 0 0; font-size: 0.9em;'>Powered by: HSV Color Moments + GLCM + LBP + SVM (class_weight='balanced')</p>
        </div>
    """)

# Launch
if __name__ == "__main__":
    print("\n" + "="*60)
    print("Starting Batik Classification Web App (20 Classes)...")
    print("="*60)
    print(f"Model: {MODEL_PATH}")
    print(f"Config: {CONFIG_PATH}")
    print(f"Classes loaded: {len(class_names)}")
    print("="*60 + "\n")

    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860
    )
