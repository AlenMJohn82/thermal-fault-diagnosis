"""
5-Fold Stratified Cross-Validation on the full 369-image thermal fault dataset.
For each fold:
  - Stage 1: Separate Physics Augmented images (filtered to train split)
  - Stage 2: Artifact-Enriched images (filtered to train split)
  - Stage 3: Clean train images
  - Evaluate on the clean held-out fold

Reports mean ± std accuracy across all 5 folds.
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

from model import PhysicsGuidedCNN
from dataset import ThermalDataset, load_dataset_paths

CLASS_MAP = {
    "A10": 0, "A30": 1, "A50": 2,
    "A&C10": 3, "A&C30": 4, "A&B50": 5,
    "A&C&B10": 6, "A&C&B30": 7,
    "Fan": 8, "Rotor-0": 9, "Noload": 10
}
CLASS_NAMES = {v: k for k, v in CLASS_MAP.items()}

BASE = "thermal ds-20260208T133253Z-1-001/thermal ds"
PATH_SEP = os.path.join(BASE, "Augmented_Separate_Physics_Dataset")
PATH_ART = os.path.join(
    BASE,
    "Augmented_Artifact_Enriched-20260224T010208Z-1-001",
    "Augmented_Artifact_Enriched"
)
PATH_CLEAN = os.path.join(BASE, "IR-Motor-bmp")

EPOCHS  = [15, 15, 8]   # [stage1, stage2, stage3] — reduced for speed
BATCH   = 32
LR      = 1e-4
IMG_SZ  = 224


def filter_paths(aug_paths, aug_labels, train_basenames, dtype):
    out_p, out_l = [], []
    for p, l in zip(aug_paths, aug_labels):
        bn = os.path.basename(p)
        parts = bn.split('_')
        if dtype == "sep" and bn.startswith('sep_') and len(parts) >= 3:
            orig = parts[1] + '.bmp'
        elif dtype == "art" and 'art' in parts and len(parts) >= 4:
            orig = parts[0] + '.bmp'
        else:
            orig = bn
        if orig in train_basenames:
            out_p.append(p)
            out_l.append(l)
    return out_p, out_l


def train_one_epoch(model, loader, opt, criterion, device):
    model.train()
    total = 0
    for imgs, masks, lbs, phys in loader:
        imgs, masks, lbs, phys = imgs.to(device), masks.to(device), lbs.to(device), phys.to(device)
        opt.zero_grad()
        out, _ = model(imgs, masks, phys)
        loss = criterion(out, lbs)
        loss.backward()
        opt.step()
        total += loss.item()
    return total / len(loader)


def run_fold(fold, train_idx, test_idx, all_paths, all_labels,
             sep_paths_all, sep_labels_all,
             art_paths_all, art_labels_all, device):

    train_paths  = [all_paths[i]  for i in train_idx]
    train_labels = [all_labels[i] for i in train_idx]
    test_paths   = [all_paths[i]  for i in test_idx]
    test_labels  = [all_labels[i] for i in test_idx]

    train_basenames = {os.path.basename(p) for p in train_paths}

    # Filter augmented sets
    sep_p, sep_l = filter_paths(sep_paths_all, sep_labels_all, train_basenames, "sep")
    art_p, art_l = filter_paths(art_paths_all, art_labels_all, train_basenames, "art")

    sep_loader   = DataLoader(ThermalDataset(sep_p, sep_l, img_size=IMG_SZ),
                              batch_size=BATCH, shuffle=True,  num_workers=2)
    art_loader   = DataLoader(ThermalDataset(art_p, art_l, img_size=IMG_SZ),
                              batch_size=BATCH, shuffle=True,  num_workers=2)
    train_loader = DataLoader(ThermalDataset(train_paths, train_labels, img_size=IMG_SZ),
                              batch_size=BATCH, shuffle=True,  num_workers=2)
    test_loader  = DataLoader(ThermalDataset(test_paths, test_labels, img_size=IMG_SZ),
                              batch_size=BATCH, shuffle=False, num_workers=2)

    model     = PhysicsGuidedCNN(num_classes=11).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # Stage 1
    print(f"  [Fold {fold}] Stage 1 — Physics Aug ({len(sep_p)} imgs)")
    for ep in range(EPOCHS[0]):
        loss = train_one_epoch(model, sep_loader, optimizer, criterion, device)
        if (ep+1) % 5 == 0:
            print(f"    Epoch {ep+1}/{EPOCHS[0]} | Loss: {loss:.4f}")
    for g in optimizer.param_groups: g["lr"] *= 0.5

    # Stage 2
    print(f"  [Fold {fold}] Stage 2 — Artifact Aug ({len(art_p)} imgs)")
    for ep in range(EPOCHS[1]):
        loss = train_one_epoch(model, art_loader, optimizer, criterion, device)
        if (ep+1) % 5 == 0:
            print(f"    Epoch {ep+1}/{EPOCHS[1]} | Loss: {loss:.4f}")
    for g in optimizer.param_groups: g["lr"] *= 0.2

    # Stage 3
    print(f"  [Fold {fold}] Stage 3 — Clean Data ({len(train_paths)} imgs)")
    for ep in range(EPOCHS[2]):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        if (ep+1) % 4 == 0:
            print(f"    Epoch {ep+1}/{EPOCHS[2]} | Loss: {loss:.4f}")

    # Evaluate
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for imgs, masks, lbs, phys in test_loader:
            imgs, masks, phys = imgs.to(device), masks.to(device), phys.to(device)
            out, _ = model(imgs, masks, phys)
            preds = torch.argmax(out, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_true.extend(lbs.numpy())

    acc = accuracy_score(all_true, all_preds)
    print(f"  [Fold {fold}] ✓ Accuracy = {acc*100:.2f}%  ({sum(p==t for p,t in zip(all_preds,all_true))}/{len(all_true)})\n")
    return acc, all_true, all_preds


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load all clean images
    all_paths, all_labels = load_dataset_paths(PATH_CLEAN, CLASS_MAP)
    all_labels_np = np.array(all_labels)
    print(f"Total images: {len(all_paths)}")

    # Load augmented datasets once
    sep_paths_all, sep_labels_all = load_dataset_paths(PATH_SEP, CLASS_MAP)
    art_paths_all, art_labels_all = load_dataset_paths(PATH_ART, CLASS_MAP)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_accs   = []
    all_preds_global = []
    all_true_global  = []

    print("\n" + "="*60)
    print("5-FOLD STRATIFIED CROSS-VALIDATION")
    print("Architecture: PG-CNN (Physics-Guided CNN)")
    print("="*60 + "\n")

    for fold, (train_idx, test_idx) in enumerate(skf.split(all_paths, all_labels_np), 1):
        print(f"{'='*40}")
        print(f"FOLD {fold}/5  |  Train: {len(train_idx)}, Test: {len(test_idx)}")
        print(f"{'='*40}")
        acc, true, preds = run_fold(
            fold, train_idx, test_idx,
            all_paths, all_labels,
            sep_paths_all, sep_labels_all,
            art_paths_all, art_labels_all,
            device
        )
        fold_accs.append(acc)
        all_preds_global.extend(preds)
        all_true_global.extend(true)

    # Final Summary
    mean_acc = np.mean(fold_accs) * 100
    std_acc  = np.std(fold_accs) * 100

    print("\n" + "="*60)
    print("FINAL CROSS-VALIDATION RESULTS")
    print("="*60)
    for i, a in enumerate(fold_accs, 1):
        print(f"  Fold {i}: {a*100:.2f}%")
    print(f"\n  Mean Accuracy : {mean_acc:.2f}%")
    print(f"  Std Deviation : ±{std_acc:.2f}%")
    print(f"\n  Statement for paper:")
    print(f"  'Our PG-CNN achieves {mean_acc:.2f}% ± {std_acc:.2f}% accuracy")
    print(f"   in 5-fold cross-validation across 369 thermal fault images.'")

    print("\nAggregate Classification Report (all folds combined):")
    print(classification_report(all_true_global, all_preds_global,
                                target_names=[CLASS_NAMES[i] for i in range(11)],
                                digits=4))

    # Save results
    results = {
        "fold_accuracies": fold_accs,
        "mean_accuracy": float(mean_acc),
        "std_accuracy": float(std_acc),
        "epochs": EPOCHS
    }
    with open("kfold_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("✓ Results saved to kfold_results.json")


if __name__ == "__main__":
    main()
