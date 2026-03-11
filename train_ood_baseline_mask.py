import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np

import torchvision.models as models
from torchvision.models import ResNet18_Weights

from dataset import ThermalDataset, load_dataset_paths
import cv2
import random
from thermal_artifacts import apply_seen_artifacts, clamp

CLASS_MAP = {
    "A10": 0, "A30": 1, "A50": 2,
    "A&C10": 3, "A&C30": 4, "A&B50": 5,
    "A&C&B10": 6, "A&C&B30": 7,
    "Fan": 8, "Rotor-0": 9, "Noload": 10
}

class BaselineMaskResNet(nn.Module):
    """Baseline ResNet18 but with simple Hotspot Mask multiplication (Guidance)"""
    def __init__(self, num_classes=11):
        super().__init__()
        backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, img, mask):
        # Extract visual features
        feats = self.backbone(img) # [B, 512, 7, 7]
        
        # Resize mask to feature map size
        mask = F.interpolate(mask, size=feats.shape[-2:], mode='nearest')
        
        # Simple Mask Guidance (Ablation isolating mask from physics)
        # We multiply features by mask
        fused_feats = feats * mask
        
        # Classification
        pooled = self.pool(fused_feats).view(img.size(0), -1)
        out = self.fc(pooled)
        return out


class OODThermalDatasetBaselineMask(ThermalDataset):
    """Dataset wrapper for baseline+mask model (Needs mask tensor)"""
    def __init__(self, image_paths, labels, img_size=224, apply_seen_ood=False):
        super().__init__(image_paths, labels, transform=None, img_size=img_size)
        self.apply_seen_ood = apply_seen_ood
        
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        img = cv2.imread(img_path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if self.apply_seen_ood:
            severity = random.randint(1, 5)
            if random.random() < 0.5:
                img_gray = apply_seen_artifacts(img_gray, severity=severity)
                img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
                
        # Generate mask
        mask_original = self.generate_hotspot_mask(img_gray)
        
        # Resize
        img_resized = cv2.resize(img, (self.img_size, self.img_size))
        mask_resized = cv2.resize(mask_original, (self.img_size, self.img_size))
        
        # Tensors
        img_tensor = torch.tensor(img_resized / 255.0).permute(2, 0, 1).float()
        mask_tensor = torch.tensor(mask_resized).unsqueeze(0).float()
        
        return img_tensor, mask_tensor, label


def filter_paths(aug_paths, aug_labels, train_basenames, dataset_type="sep"):
    filtered_paths, filtered_labels = [], []
    for path, label in zip(aug_paths, aug_labels):
        basename = os.path.basename(path)
        parts = basename.split('_')
        if dataset_type == "sep" and basename.startswith('sep_') and len(parts) >= 3:
            original_basename = parts[1] + '.bmp'
        elif dataset_type == "stoch" and 'stoch' in basename and len(parts) >= 3:
            original_basename = parts[0] + '.bmp'
        else:
            original_basename = basename
            
        if original_basename in train_basenames:
            filtered_paths.append(path)
            filtered_labels.append(label)
    return filtered_paths, filtered_labels


def train_one_epoch_mask(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for imgs, masks, labels in loader:
        imgs, masks, labels = imgs.to(device), masks.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs, masks) 
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


class BaselineMaskThermalDataset(ThermalDataset):
    def __init__(self, image_paths, labels, img_size=224):
        super().__init__(image_paths, labels, transform=None, img_size=img_size)
        
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        img = cv2.imread(img_path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask_original = self.generate_hotspot_mask(img_gray)
        
        img_resized = cv2.resize(img, (self.img_size, self.img_size))
        mask_resized = cv2.resize(mask_original, (self.img_size, self.img_size))
        
        img_tensor = torch.tensor(img_resized / 255.0).permute(2, 0, 1).float()
        mask_tensor = torch.tensor(mask_resized).unsqueeze(0).float()
        
        return img_tensor, mask_tensor, label


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    base_path = "thermal ds-20260208T133253Z-1-001/thermal ds"
    path_sep = os.path.join(base_path, "Augmented_Separate_Physics_Dataset")
    path_stoch = os.path.join(base_path, "Augmented_Combined_Stochastic")

    print("\n" + "="*60)
    print("STEP 1: REUSING OOD 70:30 SPLIT (FIXED SPLIT)")
    print("="*60)
    with open("ood_split_info.json", "r") as f:
        split_info = json.load(f)
    
    train_clean_p = split_info["train_paths"]
    path_clean = os.path.join(base_path, "IR-Motor-bmp")
    clean_paths, clean_labels = load_dataset_paths(path_clean, CLASS_MAP)
    _, _, train_clean_l, _ = train_test_split(
        clean_paths, clean_labels, test_size=0.30, stratify=clean_labels, random_state=42
    )

    train_basenames = {os.path.basename(p) for p in train_clean_p}
    print(f"Loaded FIXED untouched split (Train:{len(train_clean_p)})")

    print("\n" + "="*60)
    print("STEP 2: FILTER AUGMENTED & CREATE DATASETS (BASELINE + MASK)")
    print("="*60)
    sep_paths_all, sep_labels_all = load_dataset_paths(path_sep, CLASS_MAP)
    stoch_paths_all, stoch_labels_all = load_dataset_paths(path_stoch, CLASS_MAP)

    sep_p, sep_l = filter_paths(sep_paths_all, sep_labels_all, train_basenames, "sep")
    stoch_p, stoch_l = filter_paths(stoch_paths_all, stoch_labels_all, train_basenames, "stoch")

    train_loader_clean = DataLoader(
        OODThermalDatasetBaselineMask(train_clean_p, train_clean_l, img_size=224, apply_seen_ood=True), 
        batch_size=32, shuffle=True, num_workers=2
    )
    
    sep_loader = DataLoader(BaselineMaskThermalDataset(sep_p, sep_l, img_size=224), batch_size=32, shuffle=True, num_workers=2)
    stoch_loader = DataLoader(BaselineMaskThermalDataset(stoch_p, stoch_l, img_size=224), batch_size=32, shuffle=True, num_workers=2)

    model = BaselineMaskResNet(num_classes=11).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) 
    criterion = nn.CrossEntropyLoss()

    epochs_1, epochs_2, epochs_3 = 20, 20, 10 

    print("\n" + "="*60)
    print("STAGE 1: SEPARATE PHYSICS AUG")
    print("="*60)
    for ep in range(epochs_1):
        loss = train_one_epoch_mask(model, sep_loader, optimizer, criterion, device)
        print(f"  Epoch {ep+1}/{epochs_1} | Loss: {loss:.4f}")
    for g in optimizer.param_groups: g["lr"] *= 0.5

    print("\n" + "="*60)
    print("STAGE 2: COMBINED STOCHASTIC AUG")
    print("="*60)
    for ep in range(epochs_2):
        loss = train_one_epoch_mask(model, stoch_loader, optimizer, criterion, device)
        print(f"  Epoch {ep+1}/{epochs_2} | Loss: {loss:.4f}")
    for g in optimizer.param_groups: g["lr"] *= 0.2

    print("\n" + "="*60)
    print("STAGE 3: CLEAN IR (INJECTING SEEN ARTIFACTS DYNAMICALLY)")
    print("="*60)
    for ep in range(epochs_3):
        loss = train_one_epoch_mask(model, train_loader_clean, optimizer, criterion, device)
        print(f"  Epoch {ep+1}/{epochs_3} | Loss: {loss:.4f}")

    torch.save(model.state_dict(), "thermal_model_baseline_mask_ood_trained.pth")
    print("\n✓ Baseline+Mask OOD Model saved to thermal_model_baseline_mask_ood_trained.pth")

if __name__ == "__main__":
    main()
