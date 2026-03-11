import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np

from model import PhysicsGuidedCNN
from dataset import ThermalDataset, load_dataset_paths
import cv2
import random
from thermal_artifacts import apply_seen_artifacts

class OODThermalDataset(ThermalDataset):
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
                
        mask_original = self.generate_hotspot_mask(img_gray)
        
        img_norm = img_gray.astype(np.float32)
        img_norm = (img_norm - np.min(img_norm)) / (np.max(img_norm) + 1e-6)
        motor_mask_original = (img_norm > np.percentile(img_norm, 40)).astype(np.uint8)
        
        from dataset import extract_physics_features
        phys_feats = extract_physics_features(img_gray, mask_original, motor_mask_original)
        
        img_resized = cv2.resize(img, (self.img_size, self.img_size))
        mask_resized = cv2.resize(mask_original, (self.img_size, self.img_size))
        
        img_tensor = torch.tensor(img_resized / 255.0).permute(2, 0, 1).float()
        mask_tensor = torch.tensor(mask_resized).unsqueeze(0).float()
        phys_tensor = torch.tensor(phys_feats).float()
        
        return img_tensor, mask_tensor, label, phys_tensor

CLASS_MAP = {
    "A10": 0, "A30": 1, "A50": 2,
    "A&C10": 3, "A&C30": 4, "A&B50": 5,
    "A&C&B10": 6, "A&C&B30": 7,
    "Fan": 8, "Rotor-0": 9, "Noload": 10
}

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for imgs, masks, labels, phys in loader:
        imgs, masks, labels, phys = imgs.to(device), masks.to(device), labels.to(device), phys.to(device)
        optimizer.zero_grad()
        outputs, _ = model(imgs, masks, phys)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    base_path = "thermal ds-20260208T133253Z-1-001/thermal ds"
    path_clean = os.path.join(base_path, "IR-Motor-bmp")

    print("\n" + "="*60)
    print("STEP 1: REUSING OOD 70:30 SPLIT (FIXED SPLIT)")
    print("="*60)
    with open("ood_split_info.json", "r") as f:
        split_info = json.load(f)
    
    train_clean_p = split_info["train_paths"]
    
    # Reload original paths to get correct labels
    clean_paths, clean_labels = load_dataset_paths(path_clean, CLASS_MAP)
    _, _, train_clean_l, _ = train_test_split(
        clean_paths, clean_labels, test_size=0.30, stratify=clean_labels, random_state=42
    )

    print(f"Loaded FIXED untouched split (Train:{len(train_clean_p)})")

    # Strategy 1: "No Curriculum" - train ONLY on clean data, with NO artifacts for 50 epochs.
    print("\n" + "="*60)
    print("STRATEGY: NO CURRICULUM (CLEAN DATA ONLY)")
    print("="*60)
    
    # apply_seen_ood = False means no artifacts
    train_loader = DataLoader(
        OODThermalDataset(train_clean_p, train_clean_l, img_size=224, apply_seen_ood=False), 
        batch_size=32, shuffle=True, num_workers=2
    )

    model = PhysicsGuidedCNN(num_classes=11).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    total_epochs = 50

    for ep in range(total_epochs):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        if (ep+1) % 10 == 0 or ep == 0:
            print(f"  Epoch {ep+1}/{total_epochs} | Loss: {loss:.4f}")

    torch.save(model.state_dict(), "thermal_model_no_curr_ood_trained.pth")
    print("\n✓ No-Curriculum Model saved to thermal_model_no_curr_ood_trained.pth")

if __name__ == "__main__":
    main()
