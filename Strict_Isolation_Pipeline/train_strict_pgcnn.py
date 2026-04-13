import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import PhysicsGuidedCNN
from dataset import ThermalDataset, load_dataset_paths
from train_ood_no_curriculum import OODThermalDataset

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

    # The new fully isolated directory
    base_dir = "Strict_OOD_Pipeline/TRAIN"
    
    print("\n" + "="*60)
    print("STEP 1: LOADING ISOLATED 'TRAIN ONLY' DATASETS")
    print("="*60)
    
    # 1. Load Clean Train
    train_clean_p, train_clean_l = load_dataset_paths(os.path.join(base_dir, "Clean"), CLASS_MAP)
    print(f"Loaded Clean Train Set: {len(train_clean_p)} images")

    # 2. Load Physics Augmentations
    physics_paths, physics_labels = load_dataset_paths(os.path.join(base_dir, "Physics_Aug"), CLASS_MAP)
    print(f"Loaded Physics Augmentations: {len(physics_paths)} images")

    # 3. Load Stochastic Augmentations
    stoch_paths, stoch_labels = load_dataset_paths(os.path.join(base_dir, "Stoch_Aug"), CLASS_MAP)
    print(f"Loaded Stochastic Augmentations: {len(stoch_paths)} images")

    print("\n" + "="*60)
    print("STEP 2: CONFIGURING DATALOADERS")
    print("="*60)

    # Stage 1 loader
    sep_loader = DataLoader(ThermalDataset(physics_paths, physics_labels, img_size=224), batch_size=32, shuffle=True, num_workers=2)
    # Stage 2 loader
    stoch_loader = DataLoader(ThermalDataset(stoch_paths, stoch_labels, img_size=224), batch_size=32, shuffle=True, num_workers=2)
    # Stage 3 loader (Dynamic True because it's PG-CNN standard)
    train_loader_clean = DataLoader(
        OODThermalDataset(train_clean_p, train_clean_l, img_size=224, apply_seen_ood=True), 
        batch_size=32, shuffle=True, num_workers=2
    )

    model = PhysicsGuidedCNN(num_classes=11).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    epochs_1, epochs_2, epochs_3 = 20, 20, 10

    print("\n" + "="*60)
    print("STAGE 1: SEPARATE PHYSICS AUG REGIMEN")
    print("="*60)
    for ep in range(epochs_1):
        loss = train_one_epoch(model, sep_loader, optimizer, criterion, device)
        print(f"  Epoch {ep+1}/{epochs_1} | Loss: {loss:.4f}")
    for g in optimizer.param_groups: g["lr"] *= 0.5

    print("\n" + "="*60)
    print("STAGE 2: COMBINED STOCHASTIC AUG REGIMEN")
    print("="*60)
    for ep in range(epochs_2):
        loss = train_one_epoch(model, stoch_loader, optimizer, criterion, device)
        print(f"  Epoch {ep+1}/{epochs_2} | Loss: {loss:.4f}")
    for g in optimizer.param_groups: g["lr"] *= 0.2

    print("\n" + "="*60)
    print("STAGE 3: CLEAN IR (WITH DYNAMIC SEEN ARTIFACTS)")
    print("="*60)
    for ep in range(epochs_3):
        loss = train_one_epoch(model, train_loader_clean, optimizer, criterion, device)
        print(f"  Epoch {ep+1}/{epochs_3} | Loss: {loss:.4f}")

    torch.save(model.state_dict(), "thermal_model_strict_pipeline.pth")
    print("\n✓ Strict Pipeline Model saved to thermal_model_strict_pipeline.pth")

if __name__ == "__main__":
    main()
