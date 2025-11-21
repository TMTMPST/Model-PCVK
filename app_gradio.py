"""
Batik Classification Web App - Gradio
Upload gambar batik dan model akan mendeteksi motifnya!
"""

import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import numpy as np
import os

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading model on {device}...")

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
    best_acc = checkpoint.get('best_acc', 0)
else:
    vgg16.load_state_dict(checkpoint)
    best_acc = 0

vgg16.to(device)
vgg16.eval()

# Transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("Model loaded successfully!")
print(f"Classes: {num_classes}")
if best_acc > 0:
    print(f"Model accuracy: {best_acc:.2f}%")


def predict_batik(image):
    """Predict batik motif from uploaded image"""
    if image is None:
        return None, """
        <div style="text-align: center; padding: 3rem; color: #999;">
            <h3>📤 Upload gambar terlebih dahulu</h3>
            <p>Silakan upload atau drag & drop gambar batik</p>
        </div>
        """
    
    # Convert to PIL if needed
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    
    # Preprocess
    image_rgb = image.convert('RGB')
    input_tensor = transform(image_rgb).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = vgg16(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    # Get top 10 predictions
    topk_prob, topk_idx = torch.topk(probabilities, min(10, len(class_names)))
    
    # Format results for Gradio
    predicted_class = class_names[predicted.item()]
    confidence_score = confidence.item() * 100
    
    # Create label dict for top predictions
    predictions_dict = {
        class_names[idx.item()]: prob.item() 
        for idx, prob in zip(topk_idx[0], topk_prob[0])
    }
    
    # Extract region and pattern
    region = "Unknown"
    pattern = predicted_class
    if '_' in predicted_class:
        parts = predicted_class.split('_', 1)
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
                {predicted_class}
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
    </div>
    """
    
    return predictions_dict, result_html


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

/* Card styling */
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

/* Upload area */
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

/* Button styling */
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

/* Label styling */
.label-container {
    background: white;
    border-radius: 20px;
    padding: 1.5rem;
    box-shadow: 0 10px 40px rgba(0,0,0,0.1);
}

.label-item {
    padding: 0.8rem;
    border-radius: 10px;
    margin-bottom: 0.5rem;
    background: linear-gradient(90deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
    transition: all 0.3s ease;
}

.label-item:hover {
    background: linear-gradient(90deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
    transform: translateX(5px);
}

/* Info boxes */
.info-box {
    background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
    border-left: 4px solid #667eea;
    padding: 1.5rem;
    border-radius: 15px;
    margin: 1rem 0;
    box-shadow: 0 5px 15px rgba(0,0,0,0.05);
}

.tip-box {
    background: linear-gradient(135deg, #f093fb15 0%, #f5576c15 100%);
    border-left: 4px solid #f093fb;
    padding: 1rem;
    border-radius: 10px;
    margin: 0.5rem 0;
}

/* Markdown content styling */
.markdown-text h3 {
    color: #667eea;
    font-weight: 600;
    margin-top: 1.5rem;
    margin-bottom: 1rem;
}

.markdown-text ul {
    list-style: none;
    padding-left: 0;
}

.markdown-text li {
    padding: 0.5rem 0;
    padding-left: 1.5rem;
    position: relative;
}

.markdown-text li:before {
    content: "✓";
    color: #667eea;
    font-weight: bold;
    position: absolute;
    left: 0;
}

/* Footer */
footer {
    text-align: center;
    margin-top: 3rem;
    padding: 2rem;
    background: white;
    border-radius: 20px;
    box-shadow: 0 10px 40px rgba(0,0,0,0.1);
}

/* Examples styling */
.examples {
    margin-top: 2rem;
}

.example-image {
    border-radius: 15px;
    transition: transform 0.3s ease;
    cursor: pointer;
}

.example-image:hover {
    transform: scale(1.05);
    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
}

/* Scrollbar styling */
::-webkit-scrollbar {
    width: 10px;
}

::-webkit-scrollbar-track {
    background: #f1f1f1;
    border-radius: 10px;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 10px;
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
}

/* Animations */
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
example_folder = "data/test"
if os.path.exists(example_folder):
    # Get first image from each class (max 6 examples)
    for class_name in class_names[:6]:
        class_path = os.path.join(example_folder, class_name)
        if os.path.exists(class_path):
            images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if images:
                example_images.append(os.path.join(class_path, images[0]))

# Build Gradio interface
with gr.Blocks(css=custom_css, theme=gr.themes.Soft(
    primary_hue="purple",
    secondary_hue="blue",
    neutral_hue="gray",
    font=["Poppins", "sans-serif"]
)) as demo:
    
    # Header
    accuracy_text = f"• Accuracy: {best_acc:.1f}%" if best_acc > 0 else ""
    gr.HTML(
        f"""
        <div class="main-header">
            <h1>🎨 Batik Nusantara Classification</h1>
            <p style="font-size: 1.2rem; margin-top: 0.5rem; opacity: 0.95;">
                Deteksi Motif Batik Indonesia dengan Kecerdasan Buatan
            </p>
            <div style="margin-top: 1rem; font-size: 1rem; opacity: 0.9;">
                <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px; margin: 0 0.5rem;">
                    📚 {num_classes} Motif Batik
                </span>
                <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px; margin: 0 0.5rem;">
                    🤖 VGG16 Deep Learning
                </span>
                {f'<span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px; margin: 0 0.5rem;">🎯 {best_acc:.1f}% Accuracy</span>' if best_acc > 0 else ''}
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
    gr.HTML(f"""
        <div class="card animate-fade-in" style="margin-top: 2rem;">
            <h2 style="color: #667eea; margin-bottom: 1.5rem;">📚 Tentang Motif Batik</h2>
    
    Model ini dapat mengenali **{num_classes} motif batik** dari berbagai daerah di Indonesia:
    - **Jawa Tengah**: Truntum, Parang, Kawung, Semarangan, dll
    - **Jawa Timur**: Gentongan, Pring, dll
    - **Bali**: Barong, Merak
    - **Papua**: Asmat, Cendrawasih, Tifa
    - **Sumatra**: Boraspati, Rumah Minang
    - **Kalimantan**: Dayak, Insang
    - Dan banyak lagi...
    
    ### 🎯 Cara Menggunakan
    1. Upload gambar batik atau drag & drop ke area upload
    2. Klik tombol "Deteksi Motif Batik"
    3. Lihat hasil prediksi dengan confidence score
    4. Cek Top 10 predictions untuk alternatif motif yang mirip
    
    ### 📊 Interpretasi Hasil
    - **>90%**: Model sangat yakin dengan prediksi
    - **70-90%**: Model cukup yakin
    - **<70%**: Model kurang yakin (gambar mungkin blur atau motif tidak umum)
    """)
    
    # Footer
    gr.Markdown("""
    ---
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>🇮🇩 Batik Classification Model • VGG16 Architecture • PyTorch</p>
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
    print("Starting Gradio Web App...")
    print("="*80)
    
    demo.launch(
        share=False,  # Set True untuk public link
        server_name="0.0.0.0",  # Accessible dari network
        server_port=7860,
        show_error=True,
        inbrowser=True  # Auto open browser
    )
