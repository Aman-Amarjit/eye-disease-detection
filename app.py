import os
import io
import base64
import torch
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify, send_from_directory
from torchvision import transforms
from model import get_model
from image_enhancer import auto_enhance_eye_image

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
                        "label": "Clear Eye" if cls == "normal" else "Cataract Eye",
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
    is_auto_enhanced = False

    try:
        # Check if user requested auto-enhancement
        json_data = request.get_json(silent=True) or {}
        should_enhance = request.form.get("auto_enhance") == "true" or json_data.get("auto_enhance") is True

        # 1. Check for file upload
        if "file" in request.files and request.files["file"].filename != "":
            file = request.files["file"]
            img = Image.open(file.stream).convert("RGB")
        
        # 2. Check for base64 payload
        else:
            b64_str = json_data.get("image_base64") or request.form.get("image_base64")
            if b64_str:
                if "," in b64_str:
                    b64_str = b64_str.split(",")[1]
                img_bytes = base64.b64decode(b64_str)
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        if img is None:
            return jsonify({"error": "Please select or upload an eye photo."}), 400

        # Apply Auto-Improver (CLAHE Medical Enhancement) if enabled
        if should_enhance:
            img = auto_enhance_eye_image(img)
            is_auto_enhanced = True

        # Prepare base64 version of final image for visual preview
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        preview_b64 = "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")

        input_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            score = torch.sigmoid(output).item()

        # PyTorch ImageFolder sorts classes alphabetically: index 0 = cataract, index 1 = normal
        normal_score = score
        cataract_score = 1.0 - score

        if cataract_score >= 0.5:
            result_title = "Cataract Cloudiness Detected"
            confidence = cataract_score * 100
            user_advice = "Lens cloudiness was detected in this photo. We recommend showing this photo to an eye doctor (Ophthalmologist) for a simple checkup."
            badge_color = "#dc2626"
            badge_icon = "fa-circle-exclamation"
        else:
            result_title = "Healthy Clear Eye (No Cataract)"
            confidence = normal_score * 100
            user_advice = "Your lens appears clear and transparent. No signs of cataract cloudiness were detected in this photo."
            badge_color = "#16a34a"
            badge_icon = "fa-circle-check"

        return jsonify({
            "diagnosis": result_title,
            "confidence": round(confidence, 1),
            "cataract_probability": round(cataract_score * 100, 1),
            "normal_probability": round(normal_score * 100, 1),
            "user_advice": user_advice,
            "status_color": badge_color,
            "badge_icon": badge_icon,
            "is_auto_enhanced": is_auto_enhanced,
            "enhanced_preview_b64": preview_b64
        })

    except Exception as e:
        print(f"❌ Prediction API Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    os.makedirs("templates", exist_ok=True)
    os.makedirs("static", exist_ok=True)
    load_ai_model()
    app.run(host="0.0.0.0", port=5000, debug=True)
