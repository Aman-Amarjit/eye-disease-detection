import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from model import get_model
from data_preparation import create_sample_dataset, split_and_preprocess_dataset

def train_model(data_dir="dataset/processed", epochs=10, batch_size=16, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training device selected: {device}")
    
    # 1. Image Transformations & Normalization (MobileNetV2 Standard)
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # 2. Check if dataset exists, if not initialize
    train_path = os.path.join(data_dir, "train")
    if not os.path.exists(train_path) or len(os.listdir(train_path)) == 0:
        print("⚠️ Processed dataset not found. Initializing sample dataset...")
        create_sample_dataset()
        split_and_preprocess_dataset()

    # 3. Load Data
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ['train', 'test']
    }
    
    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=(x == 'train'), num_workers=2)
        for x in ['train', 'test']
    }

    class_names = image_datasets['train'].classes
    print(f"🏷️ Class mapping: {image_datasets['train'].class_to_idx}")
    print(f"📦 Train set size: {len(image_datasets['train'])} | Test set size: {len(image_datasets['test'])}")

    # 4. Instantiate Model, Loss Function & Optimizer
    model = get_model(freeze_base=True).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.backbone.classifier.parameters(), lr=lr)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_acc = 0.0
    start_time = time.time()

    print("\n⚡ Starting Training Phase (MobileNetV2 Transfer Learning)...")
    print("=" * 60)

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        print("-" * 30)

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            total_samples = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device).float().unsqueeze(1) # For binary BCE loss

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    preds = (torch.sigmoid(outputs) >= 0.5).float()
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()
                total_samples += inputs.size(0)

            epoch_loss = running_loss / total_samples
            epoch_acc = running_corrects / total_samples

            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc)
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), "best_cataract_mobilenet_v2.pth")

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc*100:.2f}%")

    elapsed_time = time.time() - start_time
    print("=" * 60)
    print(f"🎉 Training complete in {elapsed_time // 60:.0f}m {elapsed_time % 60:.0f}s")
    print(f"🏆 Best Test Accuracy: {best_acc * 100:.2f}%")

    # 5. Plot & Save Loss/Accuracy Training Curves
    _plot_training_history(history)
    return model, history

def _plot_training_history(history):
    epochs_range = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_loss'], label='Train Loss', color='#6366f1', linewidth=2)
    plt.plot(epochs_range, history['val_loss'], label='Test Loss', color='#f43f5e', linewidth=2)
    plt.title('Loss Curve (Binary Cross-Entropy)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, [acc*100 for acc in history['train_acc']], label='Train Acc (%)', color='#10b981', linewidth=2)
    plt.plot(epochs_range, [acc*100 for acc in history['val_acc']], label='Test Acc (%)', color='#3b82f6', linewidth=2)
    plt.title('Accuracy Curve (%)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_history.png", dpi=300)
    plt.close()
    print("📈 Saved training performance plot to 'training_history.png'")

if __name__ == "__main__":
    train_model(epochs=10, batch_size=16)
