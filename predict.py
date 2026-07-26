import os
import sys
import torch
from PIL import Image
from torchvision import transforms

from model import get_model

def predict_eye_condition(image_path, model_path="best_cataract_mobilenet_v2.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if not os.path.exists(image_path):
        print(f"❌ Error: Image file '{image_path}' not found.")
        return None

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return None

    input_tensor = transform(image).unsqueeze(0).to(device)

    model = get_model(freeze_base=True)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    with torch.no_grad():
        output = model(input_tensor)
        score = torch.sigmoid(output).item()
        
    # PyTorch ImageFolder alphabetizes classes: index 0 = cataract, index 1 = normal
    normal_prob = score
    cataract_prob = 1.0 - score

    diagnosis = "Normal Vision" if normal_prob >= 0.5 else "Cataract Detected"
    confidence = max(normal_prob, cataract_prob) * 100

    print(f"\n👁️  EYE DIAGNOSIS RESULT FOR: {os.path.basename(image_path)}")
    print("=" * 50)
    print(f"  • Diagnosis:   {diagnosis}")
    print(f"  • Confidence:  {confidence:.2f}%")
    print(f"  • Cataract Prob: {cataract_prob * 100:.2f}%")
    print(f"  • Normal Prob:   {normal_prob * 100:.2f}%")
    print("=" * 50)

    return {
        "diagnosis": diagnosis,
        "confidence": confidence,
        "cataract_prob": cataract_prob,
        "normal_prob": normal_prob
    }

if __name__ == "__main__":
    if len(sys.argv) > 1:
        predict_eye_condition(sys.argv[1])
    else:
        # Run on a sample test image
        sample_img = "dataset/processed/test/cataract/cataract_0081.jpg"
        if os.path.exists(sample_img):
            predict_eye_condition(sample_img)
        else:
            print("Usage: python predict.py <path_to_eye_image.jpg>")
