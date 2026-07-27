# 👁️ EyeCheck — AI-Powered Cataract Detection System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-MobileNetV2-EE4C2C.svg)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-Web%20App-000000.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An intuitive, high-accuracy web application designed to detect **ocular cataracts** from eye photographs and retinal fundus scans using Deep Transfer Learning and Optical Opacity Analysis.

---

## 🌟 Key Features

- 🎯 **High Clinical Accuracy (95.00%)**: Powered by an upgraded PyTorch MobileNetV2 architecture with SiLU non-linear activation and BatchNorm regularization.
- ⚡ **Instant 1-Click Detection**: Upload an eye photo or capture a live webcam snapshot to get instant results in seconds.
- ✨ **Auto-Fix Dark / Low-Contrast Photos**: Built-in OpenCV CLAHE (Contrast Limited Adaptive Histogram Equalization) image improver auto-corrects low-light or low-contrast photos.
- 🔬 **Dual Ocular Classifier**: Seamlessly handles both external eye camera photographs and retinal fundus scans.
- 💬 **Friendly Non-Intimidating UI**: Simple single-card design tailored for non-technical users, complete with clear 1-sentence health guidance.

---

## 🛠️ Tech Stack & Architecture

- **Deep Learning Framework**: [PyTorch](https://pytorch.org/) (MobileNetV2 Transfer Learning Backbone)
- **Computer Vision & Image Enhancement**: [OpenCV](https://opencv.org/) (CLAHE L-channel Contrast Equalization & Pupil Opacity Analysis)
- **Backend Web Server**: [Flask](https://flask.palletsprojects.com/) (Python)
- **Frontend Interface**: HTML5, Modern Vanilla CSS (Glassmorphism, Vibrant Palette), Responsive JS
- **Data Manipulation**: NumPy, Pillow (PIL), Torchvision

---

## 📊 Dataset & Model Performance

The model was trained strictly on an authentic dataset of **400 clinical ocular images** (300 Normal + 100 Cataract) with an 80/20 train/test split:

| Metric | Score |
|---|---|
| **Accuracy** | **95.00%** |
| **Precision** | **95.16%** |
| **Recall (Sensitivity)** | **98.33%** |
| **F1 Score** | **0.9672** |
| **ROC-AUC Score** | **0.9817** |

---

## 🚀 How Anyone Can Download & Run It (Ready Out-of-the-Box!)

> **Yes!** Anyone can download or clone this repository and run it immediately. The pre-trained model weights (`best_cataract_mobilenet_v2.pth`) are included directly in the repository, so **no re-training is required** to get exact prediction results!

### 📥 1. Clone the Repository
```bash
git clone https://github.com/Aman-Amarjit/eye-disease-detection.git
cd eye-disease-detection
```

### 📦 2. Install Dependencies
Create a virtual environment (optional but recommended) and install requirements:
```bash
python3 -m venv venv
source venv/bin/activate    # On Windows use: venv\Scripts\activate
pip install torch torchvision flask opencv-python pillow numpy requests
```

### 🌐 3. Launch the Application
```bash
python app.py
```

Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## ❓ What Happens If an Image Cannot Be Identified?

1. **Auto-Improver Feature**: If an uploaded photo is dark, blurry, or low-contrast, enable the `✨ Auto-Fix Dark / Low Contrast Photos` toggle switch before scanning. The system automatically applies adaptive contrast enhancement and sharpening to make lens cloudiness clearly visible.
2. **Safety Advisory Banner**: If an image is unclear or non-diagnostic, the app provides a friendly, non-intimidating health advice banner recommending that the user retake the photo under clear lighting or consult an eye doctor (Ophthalmologist) for a routine professional checkup.

---

## 📁 Repository Structure

```
.
├── app.py                         # Flask Web Application & Prediction API
├── model.py                       # PyTorch MobileNetV2 Model Architecture
├── image_enhancer.py              # OpenCV CLAHE Image Contrast Improver
├── factcheck_train.py             # Model Training & 6-Step Audit Pipeline
├── data_preparation.py            # Dataset Splitting & Organization Tool
├── best_cataract_mobilenet_v2.pth # Pre-trained PyTorch Model Weights
├── dataset/                       # Clinical Eye Dataset (Processed train/test)
├── sample_test_images/            # Sample thumbnails for instant testing
├── static/                        # CSS styling and static artifacts
└── templates/
    └── index.html                 # Simple, single-card web interface
```

---

## ⚖️ Disclaimer

*EyeCheck is designed as an AI-assisted preliminary screening tool and educational demonstration. It is not a substitute for professional medical diagnosis. Users should always consult a qualified Ophthalmologist for medical advice.*
