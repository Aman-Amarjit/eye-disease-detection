import os
import io
import base64
import torch
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify, send_from_directory
from torchvision import transforms
from model import get_model

app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "best_cataract_mobilenet_v2.pth"
model = None

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_ai_model():
    global model
    model = get_model(freeze_base=True)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("✅ Loaded MobileNetV2 trained model weights.")
    else:
        print("⚠️ Model checkpoint not found. Model initialized with transfer learning base.")
    model.to(device)
    model.eval()

@app.route("/")
def index():
    sample_images = []
    sample_dir = "dataset/processed/test"
    if os.path.exists(sample_dir):
        for cls in ["normal", "cataract"]:
            cls_dir = os.path.join(sample_dir, cls)
            if os.path.exists(cls_dir):
                files = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:4]
                for f in files:
                    sample_images.append({
                        "filename": f,
                        "label": cls.capitalize(),
                        "url": f"/dataset_img/test/{cls}/{f}"
                    })
    return render_template("index.html", sample_images=sample_images)

@app.route("/dataset_img/<path:filename>")
def serve_dataset_image(filename):
    return send_from_directory("dataset/processed", filename)

@app.route("/api/predict", methods=["POST"])
def predict():
    global model
    if model is None:
        load_ai_model()

    img = None

    try:
        # 1. Check for file upload in multipart/form-data
        if "file" in request.files and request.files["file"].filename != "":
            file = request.files["file"]
            img = Image.open(file.stream).convert("RGB")
        
        # 2. Check for base64 payload in JSON or Form
        else:
            json_data = request.get_json(silent=True) or {}
            b64_str = json_data.get("image_base64") or request.form.get("image_base64")
            if b64_str:
                if "," in b64_str:
                    b64_str = b64_str.split(",")[1]
                img_bytes = base64.b64decode(b64_str)
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        if img is None:
            return jsonify({"error": "No valid image provided"}), 400

        input_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            score = torch.sigmoid(output).item()

        # PyTorch ImageFolder sorts classes alphabetically: index 0 = cataract, index 1 = normal
        normal_score = score
        cataract_score = 1.0 - score

        if cataract_score >= 0.5:
            diagnosis = "CATARACT DETECTED"
            confidence = cataract_score * 100
            severity = "High Risk — Lens Opacification Identified"
            color = "#ef4444"
        else:
            diagnosis = "NORMAL CLEAR VISION"
            confidence = normal_score * 100
            severity = "Healthy Transparent Lens — No Opacity Identified"
            color = "#16a34a"

        return jsonify({
            "diagnosis": diagnosis,
            "confidence": round(confidence, 2),
            "cataract_probability": round(cataract_score * 100, 2),
            "normal_probability": round(normal_score * 100, 2),
            "severity_level": severity,
            "status_color": color
        })

    except Exception as e:
        print(f"❌ Prediction API Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/train", methods=["POST"])
def run_training_api():
    try:
        print("⚡ Triggering model training pipeline via API...")
        train_model(epochs=5, batch_size=16)
        load_ai_model()
        return jsonify({"status": "success", "message": "Training completed! Model updated."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    # Return metrics image paths or cached metrics
    return jsonify({
        "confusion_matrix_url": "/static/confusion_matrix.png" if os.path.exists("static/confusion_matrix.png") else None,
        "training_history_url": "/static/training_history.png" if os.path.exists("static/training_history.png") else None
    })

if __name__ == "__main__":
    os.makedirs("templates", exist_ok=True)
    os.makedirs("static", exist_ok=True)
    load_ai_model()
    app.run(host="0.0.0.0", port=5000, debug=True)
