"""
Batik Classification Web App - Gradio (19 Classes - Classical Features)
Upload gambar batik dan model akan mendeteksi motifnya menggunakan HSV+GLCM+LBP+SVM!
Model dengan 19 motif pilihan yang visual distinct
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
print("Loading ONNX model (19 Classes)...")

# Load the trained ONNX model
MODEL_PATH = "Batik_Group7(20C).onnx"
CONFIG_PATH = "batik_config(20C).json"

# Initialize default config
config = {
    'class_names': [],
    'IMG_SIZE': [256, 256],
    'LBP_RADIUS': 3,
    'LBP_N_POINTS': 24,
    'GLCM_DISTANCES': [1, 2],  # Multiple distances for robustness
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

def extract_glcm_features(image, distances=[1, 2], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    """Extract GLCM texture features with quantization to 32 levels

    IMPROVEMENTS (v2.0):
    - Quantize grayscale to 32 levels (instead of 256) for stability
    - Use multiple distances [1, 2] for better texture capture
    - Handle correlation NaN values gracefully
    - Average across all distances and angles for robustness
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # CRITICAL FIX: Quantize to 32 levels for stability
    # levels=256 is too large and makes GLCM sparse/noisy
    levels = 32
    quantized = (gray // (256 // levels)).astype(np.uint8)

    # Compute GLCM with reduced levels
    glcm = graycomatrix(
        quantized,
        distances=distances,
        angles=angles,
        levels=levels,
        symmetric=True,
        normed=True
    )

    # Extract properties
    properties = ['contrast', 'dissimilarity', 'homogeneity',
                  'energy', 'correlation', 'ASM']

    features = []
    for prop in properties:
        values = graycoprops(glcm, prop)

        # Average across all distances and angles
        avg_value = np.mean(values)

        # CRITICAL FIX: Handle NaN values (especially for correlation)
        if np.isnan(avg_value) or np.isinf(avg_value):
            avg_value = 0.0

        features.append(avg_value)

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
        # SVM outputs: [label, decision_values] where decision_values are NOT probabilities
        if len(outputs) >= 2:
            # Format 1: First output is label, second is probabilities/decision values
            if isinstance(outputs[0], np.ndarray) and outputs[0].dtype.kind in ('U', 'S', 'O'):
                # String labels directly
                predicted_label = str(outputs[0][0])
                raw_scores = outputs[1][0] if len(outputs) > 1 else None
            elif isinstance(outputs[1], np.ndarray) and len(outputs[1].shape) > 0:
                # Decision values in second output
                raw_scores = outputs[1][0] if len(outputs[1].shape) == 2 else outputs[1]
                predicted_idx = np.argmax(raw_scores)
                predicted_label = class_names[predicted_idx] if predicted_idx < len(class_names) else f"Class_{predicted_idx}"
            else:
                # Integer index in first output
                predicted_idx = int(outputs[0][0])
                predicted_label = class_names[predicted_idx] if predicted_idx < len(class_names) else f"Class_{predicted_idx}"
                raw_scores = outputs[1][0] if len(outputs) > 1 else None
        else:
            # Single output - could be probabilities or label
            output = outputs[0]
            if output.dtype.kind in ('U', 'S', 'O'):
                # String label
                predicted_label = str(output[0])
                raw_scores = None
            elif len(output.shape) == 2 and output.shape[1] > 1:
                # Decision values array
                raw_scores = output[0]
                predicted_idx = np.argmax(raw_scores)
                predicted_label = class_names[predicted_idx] if predicted_idx < len(class_names) else f"Class_{predicted_idx}"
            else:
                # Single integer index
                predicted_idx = int(output[0])
                predicted_label = class_names[predicted_idx] if predicted_idx < len(class_names) else f"Class_{predicted_idx}"
                raw_scores = None

        # Convert SVM decision values to probabilities using softmax
        if raw_scores is not None and len(raw_scores) > 0:
            # Apply softmax to convert decision values to probabilities
            exp_scores = np.exp(raw_scores - np.max(raw_scores))  # Subtract max for numerical stability
            probabilities = exp_scores / np.sum(exp_scores)
            confidence = float(np.max(probabilities)) * 100
        else:
            probabilities = None
            confidence = 100.0  # If no scores, assume 100% for the predicted class

        # Format prediction result with beautiful UI
        region = predicted_label.split('_')[0] if '_' in predicted_label else "Unknown"
        motif = predicted_label.split('_')[1] if '_' in predicted_label else predicted_label

        # Confidence interpretation
        if confidence >= 90:
            confidence_emoji = "🎯"
            confidence_text = "SANGAT YAKIN"
            confidence_color = "#10b981"
            confidence_desc = "Prediksi sangat akurat dan dapat dipercaya!"
        elif confidence >= 70:
            confidence_emoji = "✅"
            confidence_text = "CUKUP YAKIN"
            confidence_color = "#3b82f6"
            confidence_desc = "Prediksi cukup akurat"
        else:
            confidence_emoji = "⚠️"
            confidence_text = "KURANG YAKIN"
            confidence_color = "#f59e0b"
            confidence_desc = "Gambar mungkin blur atau motif tidak umum"

        result_html = f"""
        <div style="background: white; border-radius: 20px; padding: 2rem; box-shadow: 0 10px 40px rgba(0,0,0,0.1);">
            <div style="text-align: center; margin-bottom: 2rem;">
                <div style="font-size: 3rem; margin-bottom: 0.5rem;">{confidence_emoji}</div>
                <h2 style="color: #667eea; margin: 0; font-size: 2rem; font-weight: 700;">
                    {predicted_label}
                </h2>
            </div>

            <div style="background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
                        border-radius: 15px; padding: 1.5rem; margin-bottom: 1.5rem;">
                <div style="display: flex; justify-content: space-around; flex-wrap: wrap; gap: 1rem;">
                    <div style="text-align: center; flex: 1; min-width: 150px;">
                        <div style="color: #666; font-size: 0.9rem; margin-bottom: 0.5rem;">📍 Region</div>
                        <div style="color: #667eea; font-size: 1.3rem; font-weight: 600;">{region}</div>
                    </div>
                    <div style="text-align: center; flex: 1; min-width: 150px;">
                        <div style="color: #666; font-size: 0.9rem; margin-bottom: 0.5rem;">🎨 Pattern</div>
                        <div style="color: #764ba2; font-size: 1.3rem; font-weight: 600;">{motif}</div>
                    </div>
                </div>
            </div>

            <div style="background: linear-gradient(135deg, {confidence_color}15 0%, {confidence_color}25 100%);
                        border-left: 5px solid {confidence_color}; border-radius: 15px; padding: 1.5rem;">
                <div style="display: flex; align-items: center; gap: 1rem; flex-wrap: wrap;">
                    <div style="flex: 1; min-width: 200px;">
                        <div style="color: #666; font-size: 0.9rem; margin-bottom: 0.3rem;">Confidence Score</div>
                        <div style="color: {confidence_color}; font-size: 2rem; font-weight: 700;">
                            {confidence:.2f}%
                        </div>
                        <div style="color: {confidence_color}; font-size: 0.9rem; font-weight: 600; margin-top: 0.3rem;">
                            {confidence_text}
                        </div>
                    </div>
                    <div style="flex: 2; min-width: 250px;">
                        <div style="background: #f3f4f6; border-radius: 10px; height: 20px; overflow: hidden;">
                            <div style="background: linear-gradient(90deg, {confidence_color} 0%, {confidence_color}dd 100%);
                                        height: 100%; width: {confidence}%; transition: width 1s ease;
                                        box-shadow: 0 2px 10px {confidence_color}66;"></div>
                        </div>
                        <div style="color: #666; font-size: 0.85rem; margin-top: 0.5rem;">
                            {confidence_desc}
                        </div>
                    </div>
                </div>
            </div>

            <div style="margin-top: 1.5rem; padding: 1rem; background: #f9fafb; border-radius: 10px;">
                <div style="color: #667eea; font-weight: 600; margin-bottom: 0.5rem;">💡 Info Batik</div>
                <div style="color: #666; font-size: 0.9rem; line-height: 1.6;">
                    Motif <strong>{motif}</strong> berasal dari daerah <strong>{region}</strong>.
                    Batik ini merupakan bagian dari warisan budaya Indonesia yang kaya dan beragam.
                </div>
            </div>

            <div style="margin-top: 1rem; padding: 1rem; background: #e0f2fe; border-radius: 10px; border-left: 4px solid #0ea5e9;">
                <div style="color: #0369a1; font-weight: 600; margin-bottom: 0.5rem;">🔬 Model: Classical ML</div>
                <div style="color: #075985; font-size: 0.85rem; line-height: 1.6;">
                    Model menggunakan <strong>HSV Color Moments + GLCM Texture + LBP</strong> features dengan <strong>SVM classifier</strong>.
                    Total <strong>41 fitur</strong> diekstrak dari setiap gambar untuk klasifikasi 19 motif.
                </div>
            </div>
        </div>
        """

        # Get top 10 predictions if we have probabilities
        top_10_html = ""
        if probabilities is not None and len(class_names) > 0:
            top_10_html = """
            <div style="background: white; border-radius: 20px; padding: 2rem; box-shadow: 0 10px 40px rgba(0,0,0,0.1); margin-top: 1.5rem;">
                <h3 style="color: #667eea; margin: 0 0 1.5rem 0; font-size: 1.3rem; font-weight: 600;">
                    📊 Top 10 Predictions
                </h3>
            """

            # Get top 10 indices
            top_10_indices = np.argsort(probabilities)[-10:][::-1]

            for i, idx in enumerate(top_10_indices):
                if idx < len(class_names):
                    class_label = class_names[idx]
                    prob = probabilities[idx] * 100

                    # Color based on rank
                    if i == 0:
                        bar_color = "#667eea"
                        text_color = "#667eea"
                    elif i < 3:
                        bar_color = "#8e9eef"
                        text_color = "#764ba2"
                    else:
                        bar_color = "#b5bff0"
                        text_color = "#999"

                    top_10_html += f"""
                    <div style="margin: 1rem 0;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                            <div style="display: flex; align-items: center; gap: 0.5rem;">
                                <span style="color: {text_color}; font-weight: {'700' if i == 0 else '500'}; font-size: {'1.05rem' if i == 0 else '0.95rem'};">
                                    {class_label}
                                </span>
                            </div>
                            <span style="color: {text_color}; font-weight: 600; font-size: 0.95rem;">
                                {prob:.2f}%
                            </span>
                        </div>
                        <div style="background: #e5e7eb; border-radius: 10px; overflow: hidden; height: {'12px' if i == 0 else '10px'};">
                            <div style="background: linear-gradient(90deg, {bar_color} 0%, {bar_color}dd 100%);
                                        width: {prob}%; height: 100%; transition: width 0.8s ease;
                                        box-shadow: 0 2px 8px {bar_color}66;"></div>
                        </div>
                    </div>
                    """

            top_10_html += "</div>"

        return result_html + top_10_html, None

    except Exception as e:
        error_html = f"""
        <div style="background: white; border-radius: 20px; padding: 2rem; box-shadow: 0 10px 40px rgba(0,0,0,0.1);">
            <div style="text-align: center; margin-bottom: 1rem;">
                <div style="font-size: 3rem; margin-bottom: 0.5rem;">❌</div>
                <h2 style="color: #e74c3c; margin: 0; font-size: 1.5rem; font-weight: 700;">
                    Error Processing Image
                </h2>
            </div>
            <div style="background: #fee; border-left: 4px solid #e74c3c; padding: 1rem; border-radius: 10px;">
                <div style="color: #c0392b; font-size: 0.9rem;">
                    {str(e)}
                </div>
            </div>
        </div>
        """
        return error_html, None

# Custom CSS for beautiful UI
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

* {
    font-family: 'Poppins', sans-serif !important;
}

.gradio-container {
    max-width: 1400px !important;
    margin: auto;
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
}

.main-header {
    text-align: center;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 3rem 2rem;
    border-radius: 20px;
    margin-bottom: 2rem;
    box-shadow: 0 20px 60px rgba(102, 126, 234, 0.4);
    animation: slideDown 0.6s ease-out;
}

@keyframes slideDown {
    from {
        opacity: 0;
        transform: translateY(-30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.main-header h1 {
    font-size: 2.8rem !important;
    font-weight: 700 !important;
    margin-bottom: 1rem;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
}

.card {
    background: white;
    border-radius: 20px;
    padding: 2rem;
    box-shadow: 0 10px 40px rgba(0,0,0,0.1);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.card:hover {
    transform: translateY(-5px);
    box-shadow: 0 15px 50px rgba(0,0,0,0.15);
}

.image-container {
    background: white;
    border-radius: 20px;
    padding: 1.5rem;
    box-shadow: 0 10px 40px rgba(0,0,0,0.1);
}

.image-container img {
    border-radius: 15px;
    box-shadow: 0 5px 20px rgba(0,0,0,0.1);
}

button.primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    border: none !important;
    padding: 1rem 2rem !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    border-radius: 50px !important;
    box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4) !important;
    transition: all 0.3s ease !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}

button.primary:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 15px 40px rgba(102, 126, 234, 0.6) !important;
}

.tip-box {
    background: linear-gradient(135deg, #f093fb15 0%, #f5576c15 100%);
    border-left: 4px solid #f093fb;
    padding: 1rem;
    border-radius: 10px;
    margin: 0.5rem 0;
}

@keyframes fadeIn {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.animate-fade-in {
    animation: fadeIn 0.6s ease-out;
}
"""

# Create Gradio interface
print("Creating Gradio interface...")

with gr.Blocks(css=custom_css) as demo:

    # Header
    num_classes = len(class_names)
    gr.HTML(
        f"""
        <div class="main-header">
            <h1>🎨 Batik Nusantara Classification</h1>
            <p style="font-size: 1.2rem; margin-top: 0.5rem; opacity: 0.95;">
                Deteksi Motif Batik Indonesia dengan Classical Machine Learning
            </p>
            <div style="margin-top: 1rem; font-size: 1rem; opacity: 0.9;">
                <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px; margin: 0 0.5rem;">
                    📚 {num_classes} Motif Batik
                </span>
                <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px; margin: 0 0.5rem;">
                    🤖 HSV+GLCM+LBP+SVM
                </span>
                <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px; margin: 0 0.5rem;">
                    🔬 41 Features
                </span>
            </div>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML("""
                <div class="card animate-fade-in">
                    <h2 style="color: #667eea; margin-bottom: 1rem; font-size: 1.5rem;">
                        📤 Upload Gambar Batik
                    </h2>
                </div>
            """)

            input_image = gr.Image(
                label="",
                type="pil",
                height=400,
                elem_classes="image-container"
            )

            predict_btn = gr.Button(
                "🔍 Deteksi Motif Batik",
                variant="primary",
                size="lg",
                elem_classes="primary"
            )

            gr.HTML("""
                <div class="tip-box" style="margin-top: 1rem;">
                    <h4 style="color: #f093fb; margin: 0 0 0.5rem 0;">💡 Tips Terbaik:</h4>
                    <ul style="margin: 0; padding-left: 1.2rem; color: #666;">
                        <li>Upload gambar batik yang jelas dan fokus</li>
                        <li>Format: JPG, PNG, atau JPEG</li>
                        <li>Resolusi tinggi untuk hasil optimal</li>
                        <li>Hindari gambar blur atau terlalu gelap</li>
                        <li>Model dilatih dengan 19 motif visual distinct</li>
                    </ul>
                </div>
            """)

        with gr.Column(scale=1):
            gr.HTML("""
                <div class="card animate-fade-in">
                    <h2 style="color: #764ba2; margin-bottom: 1rem; font-size: 1.5rem;">
                        📊 Hasil Prediksi
                    </h2>
                </div>
            """)

            output_html = gr.HTML(
                """
                <div style="text-align: center; padding: 4rem 2rem; color: #999;">
                    <div style="font-size: 4rem; margin-bottom: 1rem; opacity: 0.3;">🎨</div>
                    <h3 style="color: #667eea; margin-bottom: 0.5rem;">Siap Mendeteksi!</h3>
                    <p>Upload gambar batik dan klik tombol "Deteksi Motif Batik"</p>
                </div>
                """,
                elem_classes="card"
            )

    # Available classes info
    if class_names:
        gr.HTML("""
            <div class="card animate-fade-in" style="margin-top: 2rem;">
                <h2 style="color: #667eea; margin-bottom: 1.5rem; font-size: 1.5rem;">📋 19 Motif yang Tersedia</h2>
            </div>
        """)

        classes_html = "<div class='card animate-fade-in' style='margin-top: 1rem;'>"
        classes_html += "<div style='display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 0.8rem;'>"

        for class_name in sorted(class_names):
            classes_html += f"""
            <div style='background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
                        padding: 0.8rem; border-radius: 10px; font-size: 0.9rem; color: #555;
                        border-left: 3px solid #667eea; transition: all 0.3s ease;'>
                <strong style='color: #667eea;'>•</strong> {class_name}
            </div>
            """

        classes_html += "</div></div>"
        gr.HTML(classes_html)

    # Info section
    gr.Markdown(f"""
    <div class="card animate-fade-in" style="margin-top: 2rem;">

    ## 📚 Tentang Model

    Model ini menggunakan **Classical Machine Learning** dengan ekstraksi fitur manual:
    - **HSV Color Moments**: 9 fitur (mean, std, skewness untuk H, S, V)
    - **GLCM Texture**: 6 fitur (contrast, dissimilarity, homogeneity, energy, correlation, ASM)
    - **LBP (Local Binary Pattern)**: 26 fitur (histogram uniform patterns)
    - **Classifier**: SVM dengan RBF kernel dan class_weight='balanced'

    Total **41 fitur** diekstrak dari setiap gambar untuk klasifikasi.

    ### 🎯 Cara Menggunakan
    1. Upload gambar batik atau drag & drop ke area upload
    2. Klik tombol "Deteksi Motif Batik"
    3. Lihat hasil prediksi dengan confidence score
    4. Cek Top 10 predictions untuk alternatif motif yang mirip

    ### 📊 Interpretasi Hasil
    - **>90%**: Model sangat yakin dengan prediksi
    - **70-90%**: Model cukup yakin
    - **<70%**: Model kurang yakin (gambar mungkin blur atau motif tidak umum)

    ### 🔬 Motif yang Dikenali
    Model dapat mengenali **{num_classes} motif batik** dari berbagai daerah di Indonesia

    </div>
    """)

    # Footer
    gr.Markdown("""
    ---
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>🇮🇩 Batik Classification Model • Classical ML (HSV+GLCM+LBP+SVM)</p>
        <p style="font-size: 0.9em;">Preserving Indonesian Cultural Heritage through AI</p>
        <p style="font-size: 0.85em; margin-top: 0.5rem;">PBL Kelompok 7 • Politeknik Caltex Riau</p>
    </div>
    """)

    # Connect button to prediction function
    predict_btn.click(
        fn=predict_batik,
        inputs=input_image,
        outputs=[output_html, gr.State()]
    )

# Launch
if __name__ == "__main__":
    print("\n" + "="*60)
    print("Starting Batik Classification Web App (19 Classes)...")
    print("="*60)
    print(f"Model: {MODEL_PATH}")
    print(f"Config: {CONFIG_PATH}")
    print(f"Classes loaded: {len(class_names)}")
    print("="*60 + "\n")

    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        inbrowser=True
    )
