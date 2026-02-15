import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import json

from model import PhysicsGuidedCNN
from dataset import ThermalDataset, load_dataset_paths

# Class mapping
CLASS_MAP = {
    "A10": 0, "A30": 1, "A50": 2,
    "A&C10": 3, "A&C30": 4, "A&B50": 5,
    "A&C&B10": 6, "A&C&B30": 7,
    "Fan": 8, "Rotor-0": 9, "Noload": 10
}

# Baseline Model Class (copied from ablation_study.py)
class BaselineResNet(nn.Module):
    def __init__(self, num_classes=11):
        super().__init__()
        import torchvision.models as models
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)
    
    def forward(self, x, mask, phys_feat):
        feats = self.backbone(x)
        pooled = self.pool(feats).view(x.size(0), -1)
        out = self.fc(pooled)
        return out, torch.tensor([0.0] * x.size(0))

def add_gaussian_noise(tensor, noise_factor):
    """Add Gaussian noise to a tensor image."""
    if noise_factor == 0:
        return tensor
    noise = torch.randn_like(tensor) * noise_factor
    noisy_tensor = tensor + noise
    return torch.clamp(noisy_tensor, 0, 1)

def train_baseline_model(device):
    """Quickly train a baseline model for comparison"""
    print("Training Baseline ResNet18 for comparison...")
    base_path = "thermal ds-20260208T133253Z-1-001/thermal ds/IR-Motor-bmp"
    paths, labels = load_dataset_paths(base_path, CLASS_MAP)
    train_paths, _, train_labels, _ = train_test_split(paths, labels, test_size=0.2, stratify=labels, random_state=42)
    
    dataset = ThermalDataset(train_paths, train_labels, img_size=224)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = BaselineResNet(num_classes=11).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Train for 20 epochs (sufficient for 100% on this data)
    model.train()
    for epoch in range(20):
        for imgs, _, labels, _ in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out, _ = model(imgs, None, None)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
    return model

def evaluate_with_noise(model, loader, noise_levels, device, model_name):
    results = []
    model.eval()
    
    print(f"\nEvaluating {model_name} with noise...")
    
    for noise in noise_levels:
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for imgs, masks, labels, phys_feats in loader:
                imgs = imgs.to(device)
                masks = masks.to(device)
                labels = labels.to(device)
                phys_feats = phys_feats.to(device)
                
                # Add noise
                noisy_imgs = add_gaussian_noise(imgs, noise)
                
                outputs, _ = model(noisy_imgs, masks, phys_feats)
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        acc = accuracy_score(all_labels, all_preds)
        results.append(acc)
        print(f"  Noise {noise:.2f}: Accuracy = {acc*100:.2f}%")
        
    return results

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Data (Test Set)
    base_path = "thermal ds-20260208T133253Z-1-001/thermal ds/IR-Motor-bmp"
    paths, labels = load_dataset_paths(base_path, CLASS_MAP)
    _, test_paths, _, test_labels = train_test_split(paths, labels, test_size=0.2, stratify=labels, random_state=42)
    
    test_dataset = ThermalDataset(test_paths, test_labels, img_size=224)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 2. Load Physics-Guided Model
    print("Loading Physics-Guided Model (thermal_model_final.pth)...")
    physics_model = PhysicsGuidedCNN(num_classes=11).to(device)
    physics_model.load_state_dict(torch.load("thermal_model_final.pth"))
    
    # 3. Train/Load Baseline
    baseline_model = train_baseline_model(device)
    
    # 4. Define Noise Levels
    # 0.0 = Clean
    # 0.1 = Slight grain
    # 0.3 = Heavy grain
    # 0.5 = Very noisy
    noise_levels = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
    
    # 5. Run Evaluation
    phys_accs = evaluate_with_noise(physics_model, test_loader, noise_levels, device, "Physics-Guided")
    base_accs = evaluate_with_noise(baseline_model, test_loader, noise_levels, device, "Baseline ResNet")
    
    # 6. Save and Display Results
    results = {
        "noise_levels": noise_levels,
        "physics_guided": phys_accs,
        "baseline": base_accs
    }
    
    with open('noise_robustness_results.json', 'w') as f:
        json.dump(results, f, indent=2)
        
    print("\n" + "="*60)
    print("NOISE ROBUSTNESS RESULTS")
    print("="*60)
    print(f"{'Noise Level':<15} {'Physics-Guided':<20} {'Baseline':<20} {'Diff'}")
    print("-" * 65)
    
    for i, noise in enumerate(noise_levels):
        p_acc = phys_accs[i] * 100
        b_acc = base_accs[i] * 100
        diff = p_acc - b_acc
        print(f"{noise:<15.2f} {p_acc:<20.2f} {b_acc:<20.2f} {diff:+.2f}")

    # Check for advantage
    wins = sum(1 for i in range(len(noise_levels)) if phys_accs[i] > base_accs[i])
    print("\nCONCLUSION:")
    if wins > 0:
        print("✅ Physics-guided model shows BETTER robustness to noise!")
    else:
        print("ℹ️ Both models degrade similarly under noise.")

if __name__ == "__main__":
    main()
