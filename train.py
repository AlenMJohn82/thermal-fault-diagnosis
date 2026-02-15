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
    "A10": 0,
    "A30": 1,
    "A50": 2,
    "A&C10": 3,
    "A&C30": 4,
    "A&B50": 5,
    "A&C&B10": 6,
    "A&C&B30": 7,
    "Fan": 8,
    "Rotor-0": 9,
    "Noload": 10
}

CLASS_NAMES = {v: k for k, v in CLASS_MAP.items()}


def train_one_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for imgs, masks, labels, phys_feats in loader:
        imgs = imgs.to(device)
        masks = masks.to(device)
        labels = labels.to(device)
        phys_feats = phys_feats.to(device)
        
        optimizer.zero_grad()
        outputs, alpha = model(imgs, masks, phys_feats)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for imgs, masks, labels, phys_feats in loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            labels = labels.to(device)
            phys_feats = phys_feats.to(device)
            
            outputs, alpha = model(imgs, masks, phys_feats)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy, all_preds, all_labels


def filter_augmented_paths(aug_paths, aug_labels, train_basenames):
    """
    Filter augmented dataset to only include images whose base names 
    are in the training set (exclude test images)
    
    Augmented filenames patterns: 
    - sep_032_0.bmp (Separate Physics)
    - 032_stoch_0.bmp (Stochastic)
    Original filename: 032.bmp
    """
    filtered_paths = []
    filtered_labels = []
    
    for path, label in zip(aug_paths, aug_labels):
        basename = os.path.basename(path)
        
        # Extract original image ID from augmented filename
        parts = basename.split('_')
        
        if basename.startswith('sep_') and len(parts) >= 3:
            # Pattern: sep_032_0.bmp -> 032.bmp
            original_basename = parts[1] + '.bmp'
        elif 'stoch' in basename and len(parts) >= 3:
            # Pattern: 032_stoch_0.bmp -> 032.bmp
            original_basename = parts[0] + '.bmp'
        else:
            # Fallback: use basename as-is (for clean dataset)
            original_basename = basename
        
        # Check if this original basename is in training set
        if original_basename in train_basenames:
            filtered_paths.append(path)
            filtered_labels.append(label)
    
    return filtered_paths, filtered_labels


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Define data paths
    base_path = args.data_path
    path_separate = os.path.join(base_path, "Augmented_Separate_Physics_Dataset")
    path_stochastic = os.path.join(base_path, "Augmented_Combined_Stochastic")
    path_clean = os.path.join(base_path, "IR-Motor-bmp")
    
    print("\n" + "="*50)
    print("STEP 1: SPLIT CLEAN DATA FIRST")
    print("="*50)
    
    # Load clean dataset FIRST
    clean_paths, clean_labels = load_dataset_paths(path_clean, CLASS_MAP)
    print(f"Total clean images: {len(clean_paths)}")
    
    # Split clean data (80% train, 20% test) with FIXED seed
    clean_train_paths, clean_test_paths, clean_train_labels, clean_test_labels = train_test_split(
        clean_paths,
        clean_labels,
        test_size=0.2,
        stratify=clean_labels,
        random_state=42  # Fixed seed for reproducibility
    )
    
    print(f"\nClean data split:")
    print(f"  Train: {len(clean_train_paths)} images")
    print(f"  Test: {len(clean_test_paths)} images")
    
    # Extract basenames for filtering augmented datasets
    train_basenames = {os.path.basename(p) for p in clean_train_paths}
    test_basenames = {os.path.basename(p) for p in clean_test_paths}
    
    print(f"\n✓ Training basenames: {len(train_basenames)}")
    print(f"✓ Test basenames: {len(test_basenames)}")
    
    # Save test split info
    test_split_info = {
        "test_images": clean_test_paths,
        "test_labels": [int(label) for label in clean_test_labels],
        "test_basenames": list(test_basenames),
        "class_names": CLASS_NAMES,
        "note": "These images were NEVER seen during training - not even in augmented form!"
    }
    
    test_info_file = "test_split_info.json"
    with open(test_info_file, 'w') as f:
        json.dump(test_split_info, f, indent=2)
    print(f"✓ Test split info saved to: {test_info_file}")
    
    print("\n" + "="*50)
    print("STEP 2: FILTER AUGMENTED DATASETS")
    print("="*50)
    
    # Load ALL augmented data
    sep_paths_all, sep_labels_all = load_dataset_paths(path_separate, CLASS_MAP)
    sto_paths_all, sto_labels_all = load_dataset_paths(path_stochastic, CLASS_MAP)
    
    print(f"\nBefore filtering:")
    print(f"  Separate physics: {len(sep_paths_all)} images")
    print(f"  Stochastic: {len(sto_paths_all)} images")
    
    # Filter to ONLY include training images (exclude test images)
    sep_paths, sep_labels = filter_augmented_paths(sep_paths_all, sep_labels_all, train_basenames)
    sto_paths, sto_labels = filter_augmented_paths(sto_paths_all, sto_labels_all, train_basenames)
    
    print(f"\nAfter filtering (test images removed):")
    print(f"  Separate physics: {len(sep_paths)} images")
    print(f"  Stochastic: {len(sto_paths)} images")
    
    print(f"\n✓ Test images excluded from augmented datasets!")
    print(f"✓ No data leakage - test set is completely unseen!")
    
    print("\n" + "="*50)
    print("STEP 3: CREATE DATASETS")
    print("="*50)
    
    # Create datasets
    sep_dataset = ThermalDataset(sep_paths, sep_labels, img_size=args.img_size)
    sto_dataset = ThermalDataset(sto_paths, sto_labels, img_size=args.img_size)
    clean_train_dataset = ThermalDataset(clean_train_paths, clean_train_labels, img_size=args.img_size)
    clean_test_dataset = ThermalDataset(clean_test_paths, clean_test_labels, img_size=args.img_size)
    
    # Create dataloaders
    sep_loader = DataLoader(sep_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    sto_loader = DataLoader(sto_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    clean_train_loader = DataLoader(clean_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    clean_test_loader = DataLoader(clean_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    print(f"✓ Datasets created")
    print(f"  Stage 1 loader: {len(sep_loader)} batches")
    print(f"  Stage 2 loader: {len(sto_loader)} batches")
    print(f"  Stage 3 train loader: {len(clean_train_loader)} batches")
    print(f"  Test loader: {len(clean_test_loader)} batches")
    
    # Initialize model
    model = PhysicsGuidedCNN(num_classes=11).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    print("\n" + "="*50)
    print("STAGE 1: SEPARATE PHYSICS AUGMENTATIONS")
    print("="*50)
    print(f"Training on {len(sep_paths)} augmented images (EXCLUDING test set)")
    
    for epoch in range(args.epochs_stage1):
        loss = train_one_epoch(model, sep_loader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}/{args.epochs_stage1} | Loss: {loss:.4f}")
    
    # Reduce learning rate
    for g in optimizer.param_groups:
        g["lr"] *= 0.5
    
    # Save checkpoint
    torch.save(model.state_dict(), "checkpoint_stage1.pth")
    print("✓ Stage 1 checkpoint saved")
    
    print("\n" + "="*50)
    print("STAGE 2: COMBINED STOCHASTIC AUGMENTATIONS")
    print("="*50)
    print(f"Training on {len(sto_paths)} augmented images (EXCLUDING test set)")
    
    for epoch in range(args.epochs_stage2):
        loss = train_one_epoch(model, sto_loader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}/{args.epochs_stage2} | Loss: {loss:.4f}")
    
    # Reduce learning rate again
    for g in optimizer.param_groups:
        g["lr"] *= 0.2
    
    # Save checkpoint
    torch.save(model.state_dict(), "checkpoint_stage2.pth")
    print("✓ Stage 2 checkpoint saved")
    
    print("\n" + "="*50)
    print("STAGE 3: CLEAN DATA FINE-TUNING")
    print("="*50)
    print(f"Training on {len(clean_train_paths)} clean images (EXCLUDING test set)")
    
    for epoch in range(args.epochs_stage3):
        loss = train_one_epoch(model, clean_train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}/{args.epochs_stage3} | Loss: {loss:.4f}")
    
    # Save final model
    final_model_path = "thermal_model_final.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"\n✓ Final model saved to: {final_model_path}")
    
    # Evaluate on test set
    print("\n" + "="*50)
    print("FINAL EVALUATION ON COMPLETELY UNSEEN TEST SET")
    print("="*50)
    print(f"Evaluating on {len(clean_test_paths)} test images")
    print("(These were NEVER seen - not even in augmented form!)")
    
    test_loss, test_acc, preds, labels = evaluate(model, clean_test_loader, criterion, device)
    
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    
    print("\nDetailed Classification Report:")
    print(classification_report(labels, preds, target_names=[CLASS_NAMES[i] for i in range(11)]))
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE - NO DATA LEAKAGE!")
    print("="*50)
    print(f"Final model: {final_model_path}")
    print(f"Test images list: {test_info_file}")
    print("\n✓ Test set was completely unseen in all training stages")
    print("✓ True generalization performance achieved")
    print("\nNext steps:")
    print("1. Check test_split_info.json for truly unseen test images")
    print("2. Run the web UI: python app.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train thermal fault classification model WITHOUT data leakage")
    parser.add_argument("--data_path", type=str, default="thermal ds-20260208T133253Z-1-001/thermal ds",
                        help="Path to dataset directory")
    parser.add_argument("--epochs_stage1", type=int, default=20,
                        help="Epochs for stage 1 (separate physics)")
    parser.add_argument("--epochs_stage2", type=int, default=20,
                        help="Epochs for stage 2 (stochastic)")
    parser.add_argument("--epochs_stage3", type=int, default=10,
                        help="Epochs for stage 3 (clean data)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Initial learning rate")
    parser.add_argument("--img_size", type=int, default=224,
                        help="Image size")
    
    args = parser.parse_args()
    main(args)
