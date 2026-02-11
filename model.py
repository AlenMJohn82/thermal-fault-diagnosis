import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class PhysicsReliabilityNet(nn.Module):
    """Network to predict reliability weight (alpha) from physics features"""
    
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)


def physics_guided_fusion(features, mask, alpha):
    """Fuse CNN features with physics-guided mask"""
    alpha = alpha.view(-1, 1, 1, 1)
    return alpha * (features * mask) + (1 - alpha) * features


class PhysicsGuidedCNN(nn.Module):
    """Physics-guided CNN for thermal fault classification"""
    
    def __init__(self, num_classes=11):
        super().__init__()
        
        # ResNet18 backbone (without final layers)
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        
        # Physics reliability network
        self.phys_net = PhysicsReliabilityNet()
        
        # Classifier
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)
    
    def forward(self, x, mask, phys_feat):
        # Extract features
        feats = self.backbone(x)  # [B, 512, 7, 7]
        
        # Resize mask to match feature map
        mask = F.interpolate(mask, size=feats.shape[-2:], mode='nearest')
        
        # Compute physics reliability weight
        alpha = self.phys_net(phys_feat)
        
        # Physics-guided fusion
        fused_feats = physics_guided_fusion(feats, mask, alpha)
        
        # Classification
        pooled = self.pool(fused_feats).view(x.size(0), -1)
        out = self.fc(pooled)
        
        return out, alpha
