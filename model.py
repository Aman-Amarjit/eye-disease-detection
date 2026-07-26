import torch
import torch.nn as nn
from torchvision import models

class CataractMobileNetV2(nn.Module):
    """
    MobileNetV2 Transfer Learning Architecture for Cataract Detection.
    - Base: Pretrained MobileNetV2 (ImageNet weights)
    - Base Layers: Frozen (`requires_grad = False`)
    - Classification Head: Adaptive Avg Pool -> Dropout -> Linear (Binary Output)
    """
    def __init__(self, freeze_features=True, dropout_rate=0.2):
        super(CataractMobileNetV2, self).__init__()
        
        # Load MobileNetV2 pretrained backbone
        weights = models.MobileNet_V2_Weights.DEFAULT
        self.backbone = models.mobilenet_v2(weights=weights)
        
        # Freeze base feature extractor layers
        if freeze_features:
            for param in self.backbone.features.parameters():
                param.requires_grad = False
                
        # Custom Classification Head for Binary Classification (Normal=0, Cataract=1)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, 1) # Single output for Binary Sigmoid Loss (BCEWithLogitsLoss)
        )
        
    def forward(self, x):
        return self.backbone(x)

def get_model(freeze_base=True):
    model = CataractMobileNetV2(freeze_features=freeze_base)
    return model

if __name__ == "__main__":
    model = get_model()
    print("🧠 MobileNetV2 Model Architecture for Cataract Detection initialized:")
    print(model)
    
    # Test dummy forward pass with 224x224 RGB image
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    print(f"\n✅ Input shape: {dummy_input.shape} | Output logits shape: {output.shape}")
