import os
import io
import base64
import torch
import cv2
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

def analyze_eye_type_and_opacity(img_pil):
    """
    Analyzes whether an image is a Retinal Fundus Scan or an External Eye Photo,
    and calculates lens opacity using pupil region brightness.
    
    Clinical basis:
      - Healthy clear pupil:  very dark, mean brightness < ~80
      - Cataract-affected:    lens appears whitish/grey, mean brightness > 100
    """
    img_np = np.array(img_pil.convert("RGB"))
    h, w, _ = img_np.shape

    # 1. Detect if image is a Fundus Retinal photograph (dark corners with red/orange circle)
    gray_full = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    corner_avg = float(np.mean([gray_full[0, 0], gray_full[0, -1], gray_full[-1, 0], gray_full[-1, -1]]))
    red_mean = float(np.mean(img_np[:, :, 0]))
    blue_mean = float(np.mean(img_np[:, :, 2]))
    is_fundus = (corner_avg < 25) and (red_mean > blue_mean * 1.2)

    # 2. Extract central pupil / lens region (middle 30% of width & height)
    ch_start, ch_end = int(h * 0.35), int(h * 0.65)
    cw_start, cw_end = int(w * 0.35), int(w * 0.65)
    pupil_crop = img_np[ch_start:ch_end, cw_start:cw_end]
    gray_pupil = cv2.cvtColor(pupil_crop, cv2.COLOR_RGB2GRAY)

    # Key metric: mean brightness of central pupil area
    mean_brightness = float(np.mean(gray_pupil))
    # Ratio of clearly dark pixels (healthy clear pupil)
    dark_ratio = float(np.sum(gray_pupil < 40) / gray_pupil.size)
    # Ratio of cloudy/white pixels (cataract indicator)
    cloudy_ratio = float(np.sum(gray_pupil > 120) / gray_pupil.size)

    return {
        "is_fundus": is_fundus,
        "mean_brightness": round(mean_brightness, 1),
        "dark_ratio": round(dark_ratio, 3),
        "cloudy_ratio": round(cloudy_ratio, 3),
    }

@app.route("/")
def index():
    sample_images = []
    sample_dir = "sample_test_images"
    if os.path.exists(sample_dir):
        for cls in ["normal", "cataract"]:
            cls_dir = os.path.join(sample_dir, cls)
            if os.path.exists(cls_dir):
                files = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:4]
                for f in files:
                    sample_images.append({
                        "filename": f,
                        "label": "Clear Eye" if cls == "normal" else "Cataract Eye",
                        "url": f"/sample_img/{cls}/{f}"
                    })
    return render_template("index.html", sample_images=sample_images)

@app.route("/sample_img/<path:filename>")
def serve_sample_image(filename):
    return send_from_directory("sample_test_images", filename)

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

        # Apply Auto-Improver if enabled
        if should_enhance:
            img = auto_enhance_eye_image(img)
            is_auto_enhanced = True

        # Prepare preview image
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        preview_b64 = "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Perform eye type and lens opacity analysis
        analysis = analyze_eye_type_and_opacity(img)

        # PyTorch MobileNetV2 Inference
        input_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            # PyTorch target mapping: 1.0 = Cataract, 0.0 = Normal
            sig = torch.sigmoid(output).item()
            model_cataract_score = sig

        # Combine MobileNetV2 with Pupil Opacity Analyzer
        if analysis["is_fundus"]:
            # Retinal Fundus Photo — use weighted PyTorch MobileNetV2 prediction
            cataract_score = model_cataract_score
            normal_score = 1.0 - cataract_score
        else:
            # External Outer Eye Photo — combine model score with pupil brightness/opacity metrics
            mb = analysis["mean_brightness"]
            dr = analysis["dark_ratio"]
            cr = analysis["cloudy_ratio"]

            # Definite cataract if lens is whitish/grey (mb > 95 or cr > 0.18) OR model_cataract_score > 0.40
            if mb > 95 or cr > 0.18 or model_cataract_score > 0.40:
                cataract_score = max(model_cataract_score, 0.85 if mb > 105 else 0.70)
                normal_score = 1.0 - cataract_score
            # Healthy clear eye: dark pupil (mean brightness < 75) AND strong dark ratio (> 0.20)
            elif mb < 75 and dr > 0.20:
                normal_score = 0.96
                cataract_score = 0.04
            else:
                # Borderline — use model prediction
                cataract_score = model_cataract_score
                normal_score = 1.0 - cataract_score

        if cataract_score >= 0.5:
            result_title = "Cataract Cloudiness Detected"
            confidence = cataract_score * 100
            user_advice = "Lens cloudiness was detected in this photo. We recommend showing this photo to an eye doctor (Ophthalmologist) for a simple checkup."
            badge_color = "#dc2626"
        else:
            result_title = "Healthy Clear Eye (No Cataract)"
            confidence = normal_score * 100
            user_advice = "Your eye lens appears clear and healthy! No signs of cataract cloudiness were detected in this photo."
            badge_color = "#16a34a"

        return jsonify({
            "diagnosis": result_title,
            "confidence": round(confidence, 1),
            "cataract_probability": round(cataract_score * 100, 1),
            "normal_probability": round(normal_score * 100, 1),
            "user_advice": user_advice,
            "status_color": badge_color,
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
