import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import cv2

# Import project modules
from model import PhysicsGuidedCNN
from dataset import ThermalDataset

def load_test_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data['test_images'], data['test_labels'], data['class_names']

def evaluate_model(model_path, test_images, test_labels, class_names_map, img_size=224):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load Model
    model = PhysicsGuidedCNN(num_classes=11).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Create Dataset (ThermalDataset expects lists)
    # Note: test_images in json are relative paths, need to ensure they work
    # The paths in JSON seem to start with "thermal ds-...", which is in current dir.
    
    test_dataset = ThermalDataset(test_images, test_labels, img_size=img_size)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    all_preds = []
    all_labels = []
    
    print("Running inference on test set...")
    with torch.no_grad():
        for imgs, masks, labels, phys_feats in test_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            phys_feats = phys_feats.to(device)
            
            outputs, _ = model(imgs, masks, phys_feats)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Get class names in correct order (0 to 10)
    class_names = [class_names_map[str(i)] for i in range(11)]

    # 1. Classification Report
    print("\n" + "="*60)
    print("COMPLETE CLASSIFICATION METRICS")
    print("="*60)
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    print(report)

    # 2. Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot Confusion Matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix - Thermal Fault Diagnosis')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    print("✓ Confusion Matrix plot saved to 'confusion_matrix.png'")

    # Save metrics to file
    with open('classification_metrics.txt', 'w') as f:
        f.write("="*60 + "\n")
        f.write("COMPLETE CLASSIFICATION METRICS\n")
        f.write("="*60 + "\n\n")
        f.write(report)
        f.write("\n\nConfusion Matrix:\n")
        f.write(np.array2string(cm))
    print("✓ Metrics text saved to 'classification_metrics.txt'")

if __name__ == "__main__":
    if not os.path.exists("test_split_info.json"):
        print("Error: test_split_info.json not found!")
    else:
        images, labels, class_map = load_test_data("test_split_info.json")
        evaluate_model("thermal_model_final.pth", images, labels, class_map)
