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

CLASS_MAP = {
    "A10": 0, "A30": 1, "A50": 2,
    "A&C10": 3, "A&C30": 4, "A&B50": 5,
    "A&C&B10": 6, "A&C&B30": 7,
    "Fan": 8, "Rotor-0": 9, "Noload": 10
}

class OODThermalDataset(ThermalDataset):
    def __init__(self, image_paths, labels, img_size=224, apply_seen_ood=False):
        super().__init__(image_paths, labels, transform=None, img_size=img_size)
        self.apply_seen_ood = apply_seen_ood
        
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        img = cv2.imread(img_path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # This will be bypassed in this script
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
    print("STEP 2: FILTER AUGMENTED & CREATE DATASETS")
    print("="*60)
    sep_paths_all, sep_labels_all = load_dataset_paths(path_sep, CLASS_MAP)
    stoch_paths_all, stoch_labels_all = load_dataset_paths(path_stoch, CLASS_MAP)

    sep_p, sep_l = filter_paths(sep_paths_all, sep_labels_all, train_basenames, "sep")
    stoch_p, stoch_l = filter_paths(stoch_paths_all, stoch_labels_all, train_basenames, "stoch")

    # CRITICAL CHANGE: apply_seen_ood=False for Stage 3! 
    # This means NO dynamic artifacts are injected. The model only learns from the static offline augmentations.
    train_loader_clean = DataLoader(
        OODThermalDataset(train_clean_p, train_clean_l, img_size=224, apply_seen_ood=False), 
        batch_size=32, shuffle=True, num_workers=2
    )
    
    sep_loader = DataLoader(ThermalDataset(sep_p, sep_l, img_size=224), batch_size=32, shuffle=True, num_workers=2)
    stoch_loader = DataLoader(ThermalDataset(stoch_p, stoch_l, img_size=224), batch_size=32, shuffle=True, num_workers=2)

    model = PhysicsGuidedCNN(num_classes=11).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) # Start with base LR
    criterion = nn.CrossEntropyLoss()

    epochs_1, epochs_2, epochs_3 = 20, 20, 10 # 50 epochs total

    print("\n" + "="*60)
    print("STAGE 1: SEPARATE PHYSICS AUG")
    print("="*60)
    for ep in range(epochs_1):
        loss = train_one_epoch(model, sep_loader, optimizer, criterion, device)
        print(f"  Epoch {ep+1}/{epochs_1} | Loss: {loss:.4f}")
        
    for g in optimizer.param_groups: g["lr"] *= 0.5

    print("\n" + "="*60)
    print("STAGE 2: COMBINED STOCHASTIC AUG")
    print("="*60)
    for ep in range(epochs_2):
        loss = train_one_epoch(model, stoch_loader, optimizer, criterion, device)
        print(f"  Epoch {ep+1}/{epochs_2} | Loss: {loss:.4f}")

    for g in optimizer.param_groups: g["lr"] *= 0.2

    print("\n" + "="*60)
    print("STAGE 3: CLEAN IR (NO DYNAMIC ARTIFACTS)")
    print("="*60)
    for ep in range(epochs_3):
        loss = train_one_epoch(model, train_loader_clean, optimizer, criterion, device)
        print(f"  Epoch {ep+1}/{epochs_3} | Loss: {loss:.4f}")

    torch.save(model.state_dict(), "thermal_model_static_augs_only_trained.pth")
    print("\n✓ Static Augs Only Model saved to thermal_model_static_augs_only_trained.pth")

if __name__ == "__main__":
    main()
