import os
import json
import argparse
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

from model import PhysicsGuidedCNN
from dataset import ThermalDataset, load_dataset_paths
import cv2
from thermal_artifacts import apply_seen_artifacts, clamp

# Class mapping
CLASS_MAP = {
    "A10": 0, "A30": 1, "A50": 2,
    "A&C10": 3, "A&C30": 4, "A&B50": 5,
    "A&C&B10": 6, "A&C&B30": 7,
    "Fan": 8, "Rotor-0": 9, "Noload": 10
}
CLASS_NAMES = {v: k for k, v in CLASS_MAP.items()}


class OODThermalDataset(ThermalDataset):
    def __init__(self, image_paths, labels, img_size=224, apply_seen_ood=False):
        super().__init__(image_paths, labels, transform=None, img_size=img_size)
        self.apply_seen_ood = apply_seen_ood
        
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        img = cv2.imread(img_path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply artifact to the RAW image before mask and tensor conversion
        if self.apply_seen_ood:
            # Random severity 1-5
            severity = random.randint(1, 5)
            # 50% chance to apply artifact
            if random.random() < 0.5:
                # Artifact applies to grayscale because mask expects gray
                img_gray = apply_seen_artifacts(img_gray, severity=severity)
                img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR) # Convert back to BGR structure for later steps if needed
                
        # Generate hotspot mask at original size
        mask_original = self.generate_hotspot_mask(img_gray)
        
        # Generate motor mask for physics features
        img_norm = img_gray.astype(np.float32)
        img_norm = (img_norm - np.min(img_norm)) / (np.max(img_norm) + 1e-6)
        motor_mask_original = (img_norm > np.percentile(img_norm, 40)).astype(np.uint8)
        
        # Extract physics features
        from dataset import extract_physics_features
        phys_feats = extract_physics_features(img_gray, mask_original, motor_mask_original)
        
        # Resize for CNN
        img_resized = cv2.resize(img, (self.img_size, self.img_size))
        mask_resized = cv2.resize(mask_original, (self.img_size, self.img_size))
        
        # Convert to tensors
        img_tensor = torch.tensor(img_resized / 255.0).permute(2, 0, 1).float()
        mask_tensor = torch.tensor(mask_resized).unsqueeze(0).float()
        phys_tensor = torch.tensor(phys_feats).float()
        
        if self.transform:
            img_tensor = self.transform(img_tensor)
        
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
    for imgs, masks, labels, phys_feats in loader:
        imgs, masks, labels, phys_feats = (
            imgs.to(device), masks.to(device),
            labels.to(device), phys_feats.to(device)
        )
        optimizer.zero_grad()
        outputs, _ = model(imgs, masks, phys_feats)
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
    path_clean = os.path.join(base_path, "IR-Motor-bmp")

    print("\n" + "="*60)
    print("STEP 1: OOD 70:30 SPLIT (FIXED SPLIT)")
    print("="*60)
    clean_paths, clean_labels = load_dataset_paths(path_clean, CLASS_MAP)
    
    # EXACT 70:30 fixed deterministic split
    train_clean_p, test_clean_p, train_clean_l, test_clean_l = train_test_split(
        clean_paths, clean_labels, test_size=0.30, stratify=clean_labels, random_state=42
    )
    
    train_basenames = {os.path.basename(p) for p in train_clean_p}
    
    # Save the split exactly for later OOD evaluation
    split_info = {
        "train_paths": train_clean_p,
        "test_paths": test_clean_p,
        "test_labels": [int(l) for l in test_clean_l],
        "class_names": CLASS_NAMES,
        "note": "Fixed untouched 30% test set for OOD robustness evaluation"
    }
    with open("ood_split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)
    print(f"✓ Split locked in -> ood_split_info.json (Train:{len(train_clean_p)} Test:{len(test_clean_p)})")

    print("\n" + "="*60)
    print("STEP 2: FILTER AUGMENTED & CREATE DATASETS")
    print("="*60)
    sep_paths_all, sep_labels_all = load_dataset_paths(path_sep, CLASS_MAP)
    stoch_paths_all, stoch_labels_all = load_dataset_paths(path_stoch, CLASS_MAP)

    sep_p, sep_l = filter_paths(sep_paths_all, sep_labels_all, train_basenames, "sep")
    stoch_p, stoch_l = filter_paths(stoch_paths_all, stoch_labels_all, train_basenames, "stoch")

    # The clean training set will dynamically generate SEEN artifacts!
    train_loader_clean = DataLoader(
        OODThermalDataset(train_clean_p, train_clean_l, img_size=224, apply_seen_ood=True), 
        batch_size=32, shuffle=True, num_workers=2
    )
    
    sep_loader = DataLoader(ThermalDataset(sep_p, sep_l, img_size=224), batch_size=32, shuffle=True, num_workers=2)
    stoch_loader = DataLoader(ThermalDataset(stoch_p, stoch_l, img_size=224), batch_size=32, shuffle=True, num_workers=2)

    model = PhysicsGuidedCNN(num_classes=11).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    epochs_1, epochs_2, epochs_3 = 20, 20, 10

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
    print("STAGE 3: CLEAN IR (INJECTING SEEN ARTIFACTS DYNAMICALLY)")
    print("="*60)
    for ep in range(epochs_3):
        loss = train_one_epoch(model, train_loader_clean, optimizer, criterion, device)
        print(f"  Epoch {ep+1}/{epochs_3} | Loss: {loss:.4f}")

    torch.save(model.state_dict(), "thermal_model_ood_trained.pth")
    print("\n✓ OOD-Robust Model saved to thermal_model_ood_trained.pth")
    print("Now run eval_ood.py to evaluate across 3 levels of robustness.")

if __name__ == "__main__":
    main()
