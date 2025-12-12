"""
Batik Classification Web App - Gradio (Classical Features)
Upload gambar batik dan model akan mendeteksi motifnya menggunakan HSV+GLCM+LBP+SVM!
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
print("Loading ONNX model...")

# Load the trained ONNX model
MODEL_PATH = "Batik_Group7.onnx"
CONFIG_PATH = "batik_config.json"

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
    except Exception as e:
        print(f"[WARNING] Could not load config: {e}")
else:
    print(f"[WARNING] Config file not found: {CONFIG_PATH}")
    print(f"  Using default configuration")

# Load ONNX model
if not os.path.exists(MODEL_PATH):
    print(f"[ERROR] Model file not found: {MODEL_PATH}")
    print("Please train the model first using the notebook!")
    onnx_session = None
else:
    try:
        # Create ONNX Runtime session
        onnx_session = ort.InferenceSession(MODEL_PATH)

        print(f"✓ ONNX Model loaded successfully!")
        print(f"✓ Number of classes: {len(class_names)}")

        # Get model input/output info
        input_name = onnx_session.get_inputs()[0].name
        output_name = onnx_session.get_outputs()[0].name
        print(f"✓ Model input: {input_name}")
        print(f"✓ Model output: {output_name}")

    except Exception as e:
        print(f"[ERROR] Failed to load ONNX model: {e}")
        onnx_session = None

# Configuration (default values, will be overridden if loaded from config)
IMG_SIZE = tuple(config.get('IMG_SIZE', [256, 256]))
LBP_RADIUS = config.get('LBP_RADIUS', 3)
LBP_N_POINTS = config.get('LBP_N_POINTS', 24)
GLCM_DISTANCES = config.get('GLCM_DISTANCES', [1])
GLCM_ANGLES = config.get('GLCM_ANGLES', [0, 0.785398, 1.570796, 2.356194])

print(f"\nConfiguration:")
print(f"  IMG_SIZE: {IMG_SIZE}")
print(f"  LBP: radius={LBP_RADIUS}, points={LBP_N_POINTS}")


def preprocess_image(img, target_size):
    """Preprocess image: resize and convert to different color spaces"""
    resized = cv2.resize(img, target_size)
    hsv_img = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    gray_img = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    return resized, hsv_img, gray_img


def extract_hsv_color_moments(hsv_img):
    """Extract HSV color moments (mean, std, skewness for each channel)"""
    features = []
    for channel in range(3):
        channel_data = hsv_img[:, :, channel].flatten()
        features.extend([
            np.mean(channel_data),
            np.std(channel_data),
            skew(channel_data)
        ])
    return np.array(features)


def extract_glcm_features(gray_img, distances, angles):
    """Extract GLCM texture features"""
    # Normalize to 0-255 range
    gray_normalized = cv2.normalize(gray_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Compute GLCM
    glcm = graycomatrix(
        gray_normalized,
        distances=distances,
        angles=angles,
        levels=256,
        symmetric=True,
        normed=True
    )

    # Extract properties
    properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    features = []

    for prop in properties:
        values = graycoprops(glcm, prop)
        features.append(np.mean(values))

    return np.array(features)


def extract_lbp_features(gray_img, radius, n_points):
    """Extract LBP features"""
    lbp = local_binary_pattern(gray_img, n_points, radius, method='uniform')
    n_bins = n_points + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist


def extract_features_for_image(img_bgr):
    """Extract all features (HSV + GLCM + LBP) from image"""
    # Preprocess
    _, hsv_img, gray_img = preprocess_image(img_bgr, IMG_SIZE)

    # Extract features
    hsv_features = extract_hsv_color_moments(hsv_img)
    glcm_features = extract_glcm_features(gray_img, GLCM_DISTANCES, GLCM_ANGLES)
    lbp_features = extract_lbp_features(gray_img, LBP_RADIUS, LBP_N_POINTS)

    # Combine
    combined = np.concatenate([hsv_features, glcm_features, lbp_features])

    # Handle NaN
    combined = np.nan_to_num(combined, nan=0.0)

    return combined


def predict_batik(image):
    """Predict batik motif from uploaded image"""
    if onnx_session is None:
        return None, f"""
        <div style="text-align: center; padding: 3rem; color: #e74c3c;">
            <h3>❌ ONNX Model Not Found</h3>
            <p>Please train the model first using the Jupyter Notebook!</p>
            <p style="font-size: 0.9rem; margin-top: 1rem;">
                Expected files:<br>
                <code>{MODEL_PATH}</code><br>
                <code>{CONFIG_PATH}</code>
            </p>
        </div>
        """

    if image is None:
        return None, """
        <div style="text-align: center; padding: 3rem; color: #999;">
            <h3>📤 Upload gambar terlebih dahulu</h3>
            <p>Silakan upload atau drag & drop gambar batik</p>
        </div>
        """

    try:
        # Convert to PIL if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        # Convert to BGR (OpenCV format)
        image_rgb = np.array(image.convert('RGB'))
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # Extract features
        features = extract_features_for_image(image_bgr)
        features = features.reshape(1, -1).astype(np.float32)

        # ONNX inference
        input_name = onnx_session.get_inputs()[0].name

        # Run all outputs
        output_names = [output.name for output in onnx_session.get_outputs()]
        onnx_outputs = onnx_session.run(output_names, {input_name: features})

        # Parse outputs based on what's available
        prediction_output = onnx_outputs[0]

        # Handle different output formats
        if isinstance(prediction_output[0], (list, np.ndarray)):
            # Probability output - get argmax
            probabilities = np.array(prediction_output[0])
            prediction_idx = int(np.argmax(probabilities))
        elif isinstance(prediction_output[0], str):
            # String label output
            prediction = prediction_output[0]
            prediction_idx = class_names.index(prediction) if prediction in class_names else 0
            # Create uniform probabilities
            probabilities = np.zeros(len(class_names))
            probabilities[prediction_idx] = 1.0
        else:
            # Integer index output
            prediction_idx = int(prediction_output[0])
            # Create uniform probabilities
            probabilities = np.zeros(len(class_names))
            probabilities[prediction_idx] = 1.0

        # Get prediction label
        if len(class_names) > 0 and prediction_idx < len(class_names):
            prediction = class_names[prediction_idx]
        else:
            prediction = f"Class_{prediction_idx}"

        # Get top predictions
        try:
            if len(class_names) > 0:
                top_k = min(10, len(class_names))
                top_indices = np.argsort(probabilities)[-top_k:][::-1]

                predictions_dict = {
                    class_names[idx]: float(probabilities[idx])
                    for idx in top_indices
                }

                confidence_score = probabilities[prediction_idx] * 100
            else:
                predictions_dict = {prediction: 1.0}
                confidence_score = 100.0

        except Exception as e:
            print(f"Warning: Could not get predictions dict: {e}")
            predictions_dict = {prediction: 1.0}
            confidence_score = 100.0

        # Extract region and pattern
        region = "Unknown"
        pattern = prediction
        if '_' in prediction:
            parts = prediction.split('_', 1)
            region = parts[0]
            pattern = parts[1]

        # Confidence interpretation
        if confidence_score >= 90:
            confidence_emoji = "🎯"
            confidence_text = "SANGAT YAKIN"
            confidence_color = "#10b981"
            confidence_desc = "Prediksi sangat akurat dan dapat dipercaya!"
        elif confidence_score >= 70:
            confidence_emoji = "✅"
            confidence_text = "CUKUP YAKIN"
            confidence_color = "#3b82f6"
            confidence_desc = "Prediksi cukup akurat"
        else:
            confidence_emoji = "⚠️"
            confidence_text = "KURANG YAKIN"
            confidence_color = "#f59e0b"
            confidence_desc = "Gambar mungkin blur atau motif tidak umum"

        # Create beautiful result HTML
        result_html = f"""
        <div style="background: white; border-radius: 20px; padding: 2rem; box-shadow: 0 10px 40px rgba(0,0,0,0.1);">
            <div style="text-align: center; margin-bottom: 2rem;">
                <div style="font-size: 3rem; margin-bottom: 0.5rem;">{confidence_emoji}</div>
                <h2 style="color: #667eea; margin: 0; font-size: 2rem; font-weight: 700;">
                    {prediction}
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
                        <div style="color: #764ba2; font-size: 1.3rem; font-weight: 600;">{pattern}</div>
                    </div>
                </div>
            </div>

            <div style="background: linear-gradient(135deg, {confidence_color}15 0%, {confidence_color}25 100%);
                        border-left: 5px solid {confidence_color}; border-radius: 15px; padding: 1.5rem;">
                <div style="display: flex; align-items: center; gap: 1rem; flex-wrap: wrap;">
                    <div style="flex: 1; min-width: 200px;">
                        <div style="color: #666; font-size: 0.9rem; margin-bottom: 0.3rem;">Confidence Score</div>
                        <div style="color: {confidence_color}; font-size: 2rem; font-weight: 700;">
                            {confidence_score:.2f}%
                        </div>
                        <div style="color: {confidence_color}; font-size: 0.9rem; font-weight: 600; margin-top: 0.3rem;">
                            {confidence_text}
                        </div>
                    </div>
                    <div style="flex: 2; min-width: 250px;">
                        <div style="background: #f3f4f6; border-radius: 10px; height: 20px; overflow: hidden;">
                            <div style="background: linear-gradient(90deg, {confidence_color} 0%, {confidence_color}dd 100%);
                                        height: 100%; width: {confidence_score}%; transition: width 1s ease;
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
                    Motif <strong>{pattern}</strong> berasal dari daerah <strong>{region}</strong>.
                    Batik ini merupakan bagian dari warisan budaya Indonesia yang kaya dan beragam.
                </div>
            </div>

            <div style="margin-top: 1rem; padding: 1rem; background: #e0f2fe; border-radius: 10px; border-left: 4px solid #0ea5e9;">
                <div style="color: #0369a1; font-weight: 600; margin-bottom: 0.5rem;">🔬 Model: Classical ML</div>
                <div style="color: #075985; font-size: 0.85rem; line-height: 1.6;">
                    Model menggunakan <strong>HSV Color Moments + GLCM Texture + LBP</strong> features dengan <strong>SVM classifier</strong>.
                    Total <strong>41 fitur</strong> diekstrak dari setiap gambar.
                </div>
            </div>
        </div>
        """

        return predictions_dict, result_html

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
        return None, error_html


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

# Example images (if exist)
example_images = []
example_folder = "dataset/test"
if os.path.exists(example_folder):
    # Get first image from each class (max 6 examples)
    for class_name in class_names[:6]:
        class_path = os.path.join(example_folder, class_name)
        if os.path.exists(class_path):
            images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if images:
                example_images.append(os.path.join(class_path, images[0]))

# Build Gradio interface
with gr.Blocks() as demo:

    # Inject custom CSS
    gr.HTML(f"""<style>{custom_css}</style>""")

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

            image_input = gr.Image(
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
                        <li>Format: JPG, PNG, atau GIF</li>
                        <li>Resolusi tinggi untuk hasil optimal</li>
                        <li>Hindari gambar blur atau terlalu gelap</li>
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

            result_html = gr.HTML(
                """
                <div style="text-align: center; padding: 4rem 2rem; color: #999;">
                    <div style="font-size: 4rem; margin-bottom: 1rem; opacity: 0.3;">🎨</div>
                    <h3 style="color: #667eea; margin-bottom: 0.5rem;">Siap Mendeteksi!</h3>
                    <p>Upload gambar batik dan klik tombol "Deteksi Motif Batik"</p>
                </div>
                """,
                elem_classes="card"
            )

            gr.HTML("""
                <div class="card animate-fade-in" style="margin-top: 1.5rem;">
                    <h3 style="color: #667eea; margin-bottom: 1rem;">📈 Top 10 Predictions</h3>
                </div>
            """)

            predictions_output = gr.Label(
                label="",
                num_top_classes=10,
                elem_classes="label-container"
            )

    # Examples section
    if example_images:
        gr.HTML("""
            <div class="card animate-fade-in" style="margin-top: 2rem;">
                <h2 style="color: #667eea; margin-bottom: 1.5rem;">💡 Contoh Gambar Batik</h2>
            </div>
        """)
        gr.Examples(
            examples=example_images,
            inputs=image_input,
            outputs=[predictions_output, result_html],
            fn=predict_batik,
            cache_examples=False
        )

    # Info section
    gr.Markdown(f"""
    <div class="card animate-fade-in" style="margin-top: 2rem;">

    ## 📚 Tentang Model

    Model ini menggunakan **Classical Machine Learning** dengan ekstraksi fitur manual:
    - **HSV Color Moments**: 9 fitur (mean, std, skewness untuk H, S, V)
    - **GLCM Texture**: 6 fitur (contrast, dissimilarity, homogeneity, energy, correlation, ASM)
    - **LBP (Local Binary Pattern)**: 26 fitur (histogram uniform patterns)
    - **Classifier**: SVM dengan RBF kernel

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
    </div>
    """)

    # Button action
    predict_btn.click(
        fn=predict_batik,
        inputs=image_input,
        outputs=[predictions_output, result_html]
    )


# Launch app
if __name__ == "__main__":
    print("\n" + "="*80)
    print("Starting Gradio Web App for Classical ML Batik Classification (ONNX)...")
    print("="*80)

    if onnx_session is None:
        print("\n⚠️  WARNING: ONNX Model not loaded!")
        print(f"   Please ensure '{MODEL_PATH}' and '{CONFIG_PATH}' exist.")
        print("   Train the model using the Jupyter Notebook first.\n")
    else:
        print(f"\n✓ ONNX Model loaded successfully")
        print(f"✓ Ready to classify {num_classes} batik motifs\n")

    demo.launch(
        share=False,  # Set True untuk public link
        server_name="0.0.0.0",  # Accessible dari network
        server_port=7861,  # Port berbeda dari VGG16 app
        show_error=True,
        inbrowser=True  # Auto open browser
    )
