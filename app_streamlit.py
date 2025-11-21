"""
Batik Classification Web App - Streamlit
Upload gambar batik dan model akan mendeteksi motifnya!
"""

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Page config
st.set_page_config(
    page_title="Batik Nusantara Classification",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 8px;
        border: none;
        font-size: 1.1em;
    }
    
    .prediction-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
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
        best_acc = checkpoint.get('best_acc', 0)
    else:
        vgg16.load_state_dict(checkpoint)
        best_acc = 0
    
    vgg16.to(device)
    vgg16.eval()
    
    return vgg16, class_names, device, best_acc, config

# Get transforms
def get_transforms():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# Predict function
def predict_batik(image, model, class_names, device, top_k=10):
    transform = get_transforms()
    
    # Preprocess
    image_rgb = image.convert('RGB')
    input_tensor = transform(image_rgb).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    # Get top-k predictions
    topk_prob, topk_idx = torch.topk(probabilities, min(top_k, len(class_names)))
    
    predicted_class = class_names[predicted.item()]
    confidence_score = confidence.item() * 100
    
    top_predictions = [
        {
            'class': class_names[idx.item()],
            'confidence': prob.item() * 100
        }
        for idx, prob in zip(topk_idx[0], topk_prob[0])
    ]
    
    return predicted_class, confidence_score, top_predictions

# Load model
try:
    model, class_names, device, best_acc, config = load_model()
    model_loaded = True
except Exception as e:
    st.error(f"Error loading model: {e}")
    model_loaded = False

# Header
st.markdown("""
<div class="main-header">
    <h1>🎨 Batik Nusantara Classification</h1>
    <h3>Deteksi Motif Batik Indonesia dengan AI</h3>
    <p style="font-size: 1.1em; margin-top: 10px;">
        {num_classes} Motif dari Berbagai Daerah • Powered by VGG16 Deep Learning
    </p>
</div>
""".format(num_classes=len(class_names) if model_loaded else 0), unsafe_allow_html=True)

if not model_loaded:
    st.error("⚠️ Model belum di-load. Pastikan file model ada!")
    st.stop()

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/300x150/667eea/ffffff?text=Batik+AI", use_container_width=True)
    
    st.markdown("### 📊 Model Info")
    st.metric("Total Classes", len(class_names))
    if best_acc > 0:
        st.metric("Model Accuracy", f"{best_acc:.2f}%")
    st.metric("Device", "GPU" if device.type == "cuda" else "CPU")
    
    st.markdown("---")
    st.markdown("### 🎯 Cara Menggunakan")
    st.markdown("""
    1. Upload gambar batik
    2. Tunggu hasil prediksi
    3. Lihat confidence score
    4. Cek alternatif motif
    """)
    
    st.markdown("---")
    st.markdown("### 📚 Region Coverage")
    regions = set([name.split('_')[0] for name in class_names if '_' in name])
    for region in sorted(regions)[:10]:
        st.markdown(f"• {region}")
    if len(regions) > 10:
        st.markdown(f"... dan {len(regions)-10} lainnya")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📤 Upload Gambar Batik")
    uploaded_file = st.file_uploader(
        "Pilih gambar batik (JPG, PNG, JPEG)",
        type=['jpg', 'jpeg', 'png'],
        help="Upload gambar batik yang jelas dan fokus untuk hasil terbaik"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang diupload", use_container_width=True)
        
        # Image info
        st.markdown(f"""
        **Info Gambar:**
        - Ukuran: {image.size[0]} x {image.size[1]} pixels
        - Format: {image.format}
        - Mode: {image.mode}
        """)
        
        predict_button = st.button("🔍 Deteksi Motif Batik", use_container_width=True)
    else:
        st.info("👆 Upload gambar batik untuk memulai deteksi")
        predict_button = False

with col2:
    st.markdown("### 📊 Hasil Prediksi")
    
    if uploaded_file is not None and predict_button:
        with st.spinner("🔄 Menganalisis motif batik..."):
            predicted_class, confidence_score, top_predictions = predict_batik(
                image, model, class_names, device, top_k=10
            )
        
        # Main prediction
        st.markdown(f"""
        <div class="prediction-box">
            <h2 style="color: #667eea; margin: 0;">🎨 {predicted_class}</h2>
            <h4 style="color: #666; margin-top: 0.5rem;">Confidence: {confidence_score:.2f}%</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Extract region and pattern
        if '_' in predicted_class:
            region, pattern = predicted_class.split('_', 1)
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"""
                <div class="metric-card">
                    <h4 style="color: #667eea;">📍 Region</h4>
                    <h3>{region}</h3>
                </div>
                """, unsafe_allow_html=True)
            
            with col_b:
                st.markdown(f"""
                <div class="metric-card">
                    <h4 style="color: #764ba2;">🎨 Pattern</h4>
                    <h3>{pattern}</h3>
                </div>
                """, unsafe_allow_html=True)
        
        # Confidence interpretation
        st.markdown("---")
        st.markdown("### 📈 Interpretasi Confidence")
        if confidence_score >= 90:
            st.success(f"✅ Model SANGAT YAKIN ({confidence_score:.2f}%) - Prediksi sangat akurat!")
        elif confidence_score >= 70:
            st.info(f"ℹ️ Model CUKUP YAKIN ({confidence_score:.2f}%) - Prediksi cukup akurat")
        else:
            st.warning(f"⚠️ Model KURANG YAKIN ({confidence_score:.2f}%) - Gambar mungkin blur atau motif tidak umum")
        
        # Top predictions chart
        st.markdown("---")
        st.markdown("### 📊 Top 10 Predictions")
        
        # Create dataframe
        df = pd.DataFrame(top_predictions)
        
        # Horizontal bar chart
        fig = go.Figure(go.Bar(
            x=df['confidence'],
            y=df['class'],
            orientation='h',
            marker=dict(
                color=df['confidence'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Confidence %")
            ),
            text=df['confidence'].apply(lambda x: f"{x:.2f}%"),
            textposition='outside'
        ))
        
        fig.update_layout(
            title="Top 10 Motif Predictions",
            xaxis_title="Confidence (%)",
            yaxis_title="Motif",
            height=500,
            yaxis={'categoryorder': 'total ascending'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Top predictions table
        with st.expander("📋 Lihat Detail Predictions"):
            df_display = df.copy()
            df_display['confidence'] = df_display['confidence'].apply(lambda x: f"{x:.2f}%")
            df_display.index = range(1, len(df_display) + 1)
            st.dataframe(df_display, use_container_width=True)
    
    else:
        st.info("Upload gambar dan klik tombol untuk melihat hasil prediksi")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <h3>📚 Tentang Model</h3>
    <p>Model ini menggunakan arsitektur <b>VGG16</b> yang telah di-train dengan <b>{num_classes} motif batik</b> dari berbagai daerah di Indonesia.</p>
    <p>Model dapat mengenali motif dari Jawa, Bali, Papua, Sumatra, Kalimantan, Sulawesi, dan daerah lainnya.</p>
    <br>
    <p style="font-size: 0.9em;">🇮🇩 Preserving Indonesian Cultural Heritage through AI</p>
    <p style="font-size: 0.8em; color: #999;">VGG16 Architecture • PyTorch • Deep Learning</p>
</div>
""".format(num_classes=len(class_names)), unsafe_allow_html=True)
