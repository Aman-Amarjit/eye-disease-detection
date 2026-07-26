import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix

from model import get_model

def evaluate_model(model_path="best_cataract_mobilenet_v2.pth", data_dir="dataset/processed/test"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(data_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    class_names = test_dataset.classes # ['cataract', 'normal'] or ['normal', 'cataract']
    print(f"🔍 Evaluating model on test dataset: {len(test_dataset)} images...")
    print(f"🏷️ Class Index mapping: {test_dataset.class_to_idx}")

    model = get_model(freeze_base=True)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ Model weights loaded from '{model_path}'")
    else:
        print(f"⚠️ Model file '{model_path}' not found! Evaluating randomly initialized model...")

    model.to(device)
    model.eval()

    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            preds = (probs >= 0.5).astype(int)

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_targets.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Calculate metrics
    cm = confusion_matrix(all_targets, all_preds)
    report = classification_report(all_targets, all_preds, target_names=class_names, digits=4)

    print("\n" + "="*50)
    print("📊 MODEL EVALUATION REPORT (MobileNetV2)")
    print("="*50)
    print(report)

    # Plot Confusion Matrix
    _plot_confusion_matrix(cm, class_names)
    
    return {
        'confusion_matrix': cm,
        'report': report,
        'targets': all_targets,
        'predictions': all_preds,
        'probabilities': all_probs
    }

def _plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix — Cataract Detector', fontsize=14, pad=15)
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, [c.capitalize() for c in class_names], fontsize=11)
    plt.yticks(tick_marks, [c.capitalize() for c in class_names], fontsize=11)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, f"{cm[i, j]}",
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontsize=16, fontweight="bold")

    plt.ylabel('Actual Ground Truth', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300)
    plt.close()
    print("🖼️ Saved confusion matrix graphic to 'confusion_matrix.png'")

if __name__ == "__main__":
    evaluate_model()
