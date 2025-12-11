# 🎨 Batik Classification - Gradio Web App (ONNX)

Aplikasi web untuk klasifikasi motif batik menggunakan Classical Machine Learning (HSV + GLCM + LBP + SVM) dengan ONNX Runtime.

## 📋 Prerequisites

Pastikan kamu sudah:

1. ✅ Melatih model menggunakan notebook `Batik_Classification_HSV_GLCM_SVM.ipynb`
2. ✅ File ONNX model `Batik_Group7.onnx` sudah tersedia
3. ✅ File konfigurasi `batik_config.json` sudah tersedia

## 🚀 Cara Menjalankan

### 1. Install Dependencies

```bash
pip install gradio opencv-python scikit-image scipy onnxruntime numpy pillow
```

### 2. Pastikan Model & Config Tersedia

Jalankan notebook sampai bagian:

- **"ONNX Model Export"** untuk generate `Batik_Group7.onnx`
- **"Save Configuration for Gradio App"** untuk generate `batik_config.json`

### 3. Jalankan Aplikasi

```bash
python app_gradio_classical.py
```

Aplikasi akan otomatis membuka browser di `http://localhost:7861`

## 🎯 Fitur

- **Upload Gambar**: Drag & drop atau klik untuk upload gambar batik
- **Prediksi Real-time**: Deteksi motif batik secara instant
- **Confidence Score**: Melihat tingkat keyakinan model
- **Top 10 Predictions**: Alternatif prediksi motif yang mirip
- **Info Regional**: Menampilkan asal daerah motif batik
- **Beautiful UI**: Interface modern dengan animasi smooth

## 🔬 Teknologi

- **Model Format**: ONNX (Open Neural Network Exchange) - portable & framework-independent
- **Runtime**: ONNX Runtime untuk inferensi cepat
- **Features**: HSV Color Moments (9) + GLCM Texture (6) + LBP (26) = **41 fitur**
- **Classifier**: SVM dengan RBF kernel
- **Preprocessing**: PCA untuk dimensionality reduction
- **Normalization**: StandardScaler untuk feature scaling

### ✨ Kenapa ONNX?

- ✅ **Portable**: Tidak tergantung pada scikit-learn version
- ✅ **Fast**: ONNX Runtime dioptimasi untuk inferensi
- ✅ **Standard**: Format industri untuk model deployment
- ✅ **Compatible**: Bisa digunakan di berbagai platform (Python, C++, JavaScript, dll)

## 📊 Interpretasi Hasil

### Confidence Score

- **90-100%** 🎯: Model sangat yakin (prediksi akurat)
- **70-90%** ✅: Model cukup yakin (prediksi dapat dipercaya)
- **<70%** ⚠️: Model kurang yakin (gambar mungkin blur atau motif tidak umum)

### Tips Upload Gambar

1. Gunakan gambar dengan **resolusi tinggi**
2. Pastikan gambar **fokus pada motif batik**
3. Hindari gambar yang **blur atau gelap**
4. Format yang didukung: **JPG, PNG, GIF**

## 🎨 Motif yang Dikenali

Model dapat mengenali berbagai motif batik dari seluruh Indonesia:

- **Jawa Tengah**: Truntum, Parang, Kawung, Semarangan, Sidoluhur, dll
- **Jawa Timur**: Gentongan, Pring
- **Yogyakarta**: Parang Barong, Kawung, Ceplok Liring, dll
- **Bali**: Barong, Merak
- **Papua**: Asmat, Cendrawasih, Tifa
- **Sumatra**: Boraspati, Rumah Minang, Pintu Aceh
- **Kalimantan**: Dayak, Insang
- **Lampung**: Gajah, Bledheg, Kacang Hijau
- Dan banyak lagi...

## 🔧 Troubleshooting

### ONNX Model Not Found Error

**Masalah**: Aplikasi menampilkan "ONNX Model Not Found"

**Solusi**:

1. Pastikan file `Batik_Group7.onnx` dan `batik_config.json` ada di direktori yang sama
2. Jalankan notebook sampai cell "ONNX Model Export" dan "Save Configuration for Gradio App"
3. Cek apakah kedua file berhasil di-generate

### Port Already in Use

**Masalah**: Error "Address already in use"

**Solusi**:

```bash
# Ganti port di app_gradio_classical.py (line terakhir)
server_port=7862  # Ganti dengan port lain
```

### Import Error

**Masalah**: ModuleNotFoundError

**Solusi**:

```bash
# Install semua dependencies
pip install -r requirements.txt
```

## 📝 File Structure

```
.
├── app_gradio_classical.py       # Aplikasi Gradio (ONNX Runtime)
├── Batik_Group7.onnx            # ONNX model (generate dari notebook)
├── batik_config.json            # Konfigurasi model (generate dari notebook)
├── Batik_Classification_HSV_GLCM_SVM.ipynb  # Notebook training
├── dataset/                     # Dataset batik
│   ├── train/
│   ├── test/
│   └── val/
└── README_GRADIO.md             # Dokumentasi ini
```

## 🌐 Network Access

Aplikasi berjalan di `0.0.0.0:7861`, artinya bisa diakses dari:

- **Localhost**: `http://localhost:7861`
- **Network**: `http://<IP-komputer-kamu>:7861`

Untuk mendapatkan public link (share ke internet):

```python
# Edit di app_gradio_classical.py
demo.launch(
    share=True,  # Ubah ke True
    ...
)
```

## 📞 Support

Jika ada masalah atau pertanyaan, silakan cek:

1. Notebook `Batik_Classification_HSV_GLCM_SVM.ipynb`
2. Requirements di `requirements.txt`
3. Logs di terminal saat menjalankan aplikasi

---

**🇮🇩 Preserving Indonesian Cultural Heritage through AI**
