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


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Define data paths
    base_path = args.data_path
    path_separate = os.path.join(base_path, "Augmented_Separate_Physics_Dataset")
    path_stochastic = os.path.join(base_path, "Augmented_Combined_Stochastic")
    path_clean = os.path.join(base_path, "IR-Motor-bmp")
    
    print("\n" + "="*50)
    print("LOADING DATASETS")
    print("="*50)
    
    # Load curriculum datasets
    sep_paths, sep_labels = load_dataset_paths(path_separate, CLASS_MAP)
    sto_paths, sto_labels = load_dataset_paths(path_stochastic, CLASS_MAP)
    clean_paths, clean_labels = load_dataset_paths(path_clean, CLASS_MAP)
    
    print(f"Separate Physics: {len(sep_paths)} images")
    print(f"Stochastic: {len(sto_paths)} images")
    print(f"Clean: {len(clean_paths)} images")
    
    # Split clean data (80% train, 20% test) with FIXED seed for reproducibility
    clean_train_paths, clean_test_paths, clean_train_labels, clean_test_labels = train_test_split(
        clean_paths,
clean_labels,
        test_size=0.2,
        stratify=clean_labels,
        random_state=42  # Fixed seed!
    )
    
    print(f"\nClean split:")
    print(f"  Train: {len(clean_train_paths)} images")
    print(f"  Test: {len(clean_test_paths)} images")
    
    # Save test split info for user
    test_split_info = {
        "test_images": clean_test_paths,
        "test_labels": [int(label) for label in clean_test_labels],
        "class_names": CLASS_NAMES,
        "note": "These images were NOT seen during training - use them for inference testing"
    }
    
    test_info_file = "test_split_info.json"
    with open(test_info_file, 'w') as f:
        json.dump(test_split_info, f, indent=2)
    print(f"\n✓ Test split info saved to: {test_info_file}")
    
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
    
    # Initialize model
    model = PhysicsGuidedCNN(num_classes=11).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    print("\n" + "="*50)
    print("STAGE 1: SEPARATE PHYSICS AUGMENTATIONS")
    print("="*50)
    
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
    
    for epoch in range(args.epochs_stage3):
        loss = train_one_epoch(model, clean_train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}/{args.epochs_stage3} | Loss: {loss:.4f}")
    
    # Save final model
    final_model_path = "thermal_model_final.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"\n✓ Final model saved to: {final_model_path}")
    
    # Evaluate on test set
    print("\n" + "="*50)
    print("FINAL EVALUATION ON TEST SET")
    print("="*50)
    
    test_loss, test_acc, preds, labels = evaluate(model, clean_test_loader, criterion, device)
    
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    
    print("\nDetailed Classification Report:")
    print(classification_report(labels, preds, target_names=[CLASS_NAMES[i] for i in range(11)]))
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE!")
    print("="*50)
    print(f"Final model: {final_model_path}")
    print(f"Test images list: {test_info_file}")
    print("\nNext steps:")
    print("1. Check test_split_info.json for images to use for inference testing")
    print("2. Run the web UI: python app.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train thermal fault classification model")
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
