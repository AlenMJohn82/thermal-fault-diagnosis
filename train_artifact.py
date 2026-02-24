import os
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

from model import PhysicsGuidedCNN
from dataset import ThermalDataset, load_dataset_paths


# Class mapping
CLASS_MAP = {
    "A10": 0, "A30": 1, "A50": 2,
    "A&C10": 3, "A&C30": 4, "A&B50": 5,
    "A&C&B10": 6, "A&C&B30": 7,
    "Fan": 8, "Rotor-0": 9, "Noload": 10
}
CLASS_NAMES = {v: k for k, v in CLASS_MAP.items()}


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for imgs, masks, labels, phys_feats in loader:
        imgs, masks, labels, phys_feats = (
            imgs.to(device), masks.to(device),
            labels.to(device), phys_feats.to(device)
        )
        optimizer.zero_grad()
        outputs, alpha = model(imgs, masks, phys_feats)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, masks, labels, phys_feats in loader:
            imgs, masks, labels, phys_feats = (
                imgs.to(device), masks.to(device),
                labels.to(device), phys_feats.to(device)
            )
            outputs, _ = model(imgs, masks, phys_feats)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return total_loss / len(loader), accuracy_score(all_labels, all_preds), all_preds, all_labels


def filter_paths(aug_paths, aug_labels, train_basenames, dataset_type="sep"):
    """
    Filter augmented paths to only include images whose source files
    are in the training set (not the test set).

    Filename patterns:
      - Separate Physics: sep_{id}_{n}.bmp  -> original: {id}.bmp
      - Artifact:         {id}_art_{n}_{augtype}.bmp -> original: {id}.bmp
    """
    filtered_paths, filtered_labels = [], []
    for path, label in zip(aug_paths, aug_labels):
        basename = os.path.basename(path)
        parts = basename.split('_')

        if dataset_type == "sep" and basename.startswith('sep_') and len(parts) >= 3:
            # sep_032_0.bmp -> 032.bmp
            original_basename = parts[1] + '.bmp'
        elif dataset_type == "art" and 'art' in parts and len(parts) >= 4:
            # 032_art_0_hotspot.bmp -> 032.bmp
            original_basename = parts[0] + '.bmp'
        else:
            original_basename = basename  # fallback for clean data

        if original_basename in train_basenames:
            filtered_paths.append(path)
            filtered_labels.append(label)
    return filtered_paths, filtered_labels


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    base_path = args.data_path
    path_separate = os.path.join(base_path, "Augmented_Separate_Physics_Dataset")
    path_artifact = os.path.join(
        base_path,
        "Augmented_Artifact_Enriched-20260224T010208Z-1-001",
        "Augmented_Artifact_Enriched"
    )
    path_clean = os.path.join(base_path, "IR-Motor-bmp")

    print("\n" + "="*60)
    print("STEP 1: SPLIT CLEAN DATA (70% Train / 30% Test)")
    print("="*60)

    clean_paths, clean_labels = load_dataset_paths(path_clean, CLASS_MAP)
    print(f"Total clean images: {len(clean_paths)}")

    # 70:30 split
    clean_train_paths, clean_test_paths, clean_train_labels, clean_test_labels = train_test_split(
        clean_paths, clean_labels,
        test_size=0.30,
        stratify=clean_labels,
        random_state=42
    )
    print(f"  Train: {len(clean_train_paths)} images")
    print(f"  Test : {len(clean_test_paths)} images")

    train_basenames = {os.path.basename(p) for p in clean_train_paths}
    test_basenames  = {os.path.basename(p) for p in clean_test_paths}

    # Save test split info (separate file so original is preserved)
    test_split_info = {
        "test_images":    clean_test_paths,
        "test_labels":    [int(l) for l in clean_test_labels],
        "test_basenames": list(test_basenames),
        "class_names":    CLASS_NAMES,
        "note":           "70:30 split – Artifact-enriched Stage 2 experiment"
    }
    with open("test_split_info_artifact.json", 'w') as f:
        json.dump(test_split_info, f, indent=2)
    print("✓ Test split saved to test_split_info_artifact.json")

    print("\n" + "="*60)
    print("STEP 2: FILTER AUGMENTED DATASETS")
    print("="*60)

    sep_paths_all, sep_labels_all = load_dataset_paths(path_separate, CLASS_MAP)
    art_paths_all, art_labels_all = load_dataset_paths(path_artifact, CLASS_MAP)

    print(f"Before filtering:")
    print(f"  Separate Physics : {len(sep_paths_all)} images")
    print(f"  Artifact Enriched: {len(art_paths_all)} images")

    sep_paths, sep_labels = filter_paths(sep_paths_all, sep_labels_all, train_basenames, "sep")
    art_paths, art_labels = filter_paths(art_paths_all, art_labels_all, train_basenames, "art")

    print(f"After filtering (test images removed):")
    print(f"  Separate Physics : {len(sep_paths)} images")
    print(f"  Artifact Enriched: {len(art_paths)} images")

    print("\n" + "="*60)
    print("STEP 3: CREATE DATASETS & DATALOADERS")
    print("="*60)

    sep_dataset   = ThermalDataset(sep_paths, sep_labels, img_size=args.img_size)
    art_dataset   = ThermalDataset(art_paths, art_labels, img_size=args.img_size)
    train_dataset = ThermalDataset(clean_train_paths, clean_train_labels, img_size=args.img_size)
    test_dataset  = ThermalDataset(clean_test_paths,  clean_test_labels,  img_size=args.img_size)

    sep_loader   = DataLoader(sep_dataset,   batch_size=args.batch_size, shuffle=True,  num_workers=2)
    art_loader   = DataLoader(art_dataset,   batch_size=args.batch_size, shuffle=True,  num_workers=2)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  num_workers=2)
    test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False, num_workers=2)
    print("✓ All dataloaders ready")

    # Model
    model     = PhysicsGuidedCNN(num_classes=11).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # ── STAGE 1: Separate Physics Augmentations ───────────────────────────
    print("\n" + "="*60)
    print("STAGE 1: SEPARATE PHYSICS AUGMENTATIONS")
    print("="*60)
    for epoch in range(args.epochs_stage1):
        loss = train_one_epoch(model, sep_loader, optimizer, criterion, device)
        print(f"  Epoch {epoch+1}/{args.epochs_stage1} | Loss: {loss:.4f}")
    for g in optimizer.param_groups:
        g["lr"] *= 0.5
    torch.save(model.state_dict(), "checkpoint_artifact_stage1.pth")
    print("✓ Stage 1 checkpoint saved")

    # ── STAGE 2: Artifact-Enriched Augmentations (replaces Stochastic) ───
    print("\n" + "="*60)
    print("STAGE 2: ARTIFACT-ENRICHED AUGMENTATIONS (NEW)")
    print("="*60)
    for epoch in range(args.epochs_stage2):
        loss = train_one_epoch(model, art_loader, optimizer, criterion, device)
        print(f"  Epoch {epoch+1}/{args.epochs_stage2} | Loss: {loss:.4f}")
    for g in optimizer.param_groups:
        g["lr"] *= 0.2
    torch.save(model.state_dict(), "checkpoint_artifact_stage2.pth")
    print("✓ Stage 2 checkpoint saved")

    # ── STAGE 3: Clean Data Fine-Tuning ──────────────────────────────────
    print("\n" + "="*60)
    print("STAGE 3: CLEAN DATA FINE-TUNING")
    print("="*60)
    for epoch in range(args.epochs_stage3):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"  Epoch {epoch+1}/{args.epochs_stage3} | Loss: {loss:.4f}")

    # Save separately so the original model is NOT overwritten
    final_model_path = "thermal_model_artifact.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"\n✓ Final model saved → {final_model_path}")

    # ── FINAL EVALUATION ──────────────────────────────────────────────────
    print("\n" + "="*60)
    print("FINAL EVALUATION (30% held-out test set)")
    print("="*60)
    test_loss, test_acc, preds, labels = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss    : {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print("\nDetailed Classification Report:")
    print(classification_report(labels, preds, target_names=[CLASS_NAMES[i] for i in range(11)]))

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"✓ New model  : {final_model_path}")
    print(f"✓ Old model  : thermal_model_final.pth")
    print("  → Run noise_test.py on both to compare robustness!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train artifact-enriched PG-CNN (70:30 split)")
    parser.add_argument("--data_path",      type=str,   default="thermal ds-20260208T133253Z-1-001/thermal ds")
    parser.add_argument("--epochs_stage1",  type=int,   default=20)
    parser.add_argument("--epochs_stage2",  type=int,   default=20)
    parser.add_argument("--epochs_stage3",  type=int,   default=10)
    parser.add_argument("--batch_size",     type=int,   default=32)
    parser.add_argument("--lr",             type=float, default=1e-4)
    parser.add_argument("--img_size",       type=int,   default=224)
    args = parser.parse_args()
    main(args)
