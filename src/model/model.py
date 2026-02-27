# src/models/model.py
import torch
import torch.nn as nn
import timm

class TeaLeafClassifier(nn.Module):
    def __init__(self, model_name='efficientnet_b0', num_classes=4, pretrained=True, drop_rate=0.5):
        """
        Initializes the Tea Leaf Classification model.
        
        Args:
            model_name (str): The name of the model architecture in timm.
            num_classes (int): Number of output classes (T1, T2, T3, T4).
            pretrained (bool): Whether to use ImageNet pre-trained weights.
            drop_rate (float): Dropout rate before classifier head.
        """
        super(TeaLeafClassifier, self).__init__()
        
        print(f"Initializing {model_name} (dropout={drop_rate})...")
        
        # Load backbone WITHOUT the default classifier head (num_classes=0 returns features)
        self.backbone = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            num_classes=0  # Remove default head to add our own with dropout
        )
        
        # Get the number of features from the backbone
        num_features = self.backbone.num_features
        
        # Custom classifier head with dropout for regularization
        self.head = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        """
        Forward pass of the model.
        """
        features = self.backbone(x)
        output = self.head(features)

        return output

def create_model(model_name='efficientnet_b0', num_classes=4, pretrained=True):
    """
    Factory function to instantiate the model.
    """
    model = TeaLeafClassifier(
        model_name=model_name, 
        num_classes=num_classes, 
        pretrained=pretrained
    )
    return model