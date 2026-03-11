import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

from train_ood_baseline_static import StaticFusionResNet, StaticFusionThermalDataset

import cv2
from thermal_artifacts import (
    add_stripe_noise, add_gradient_drift, inject_local_hotspot, global_thermal_bias,
    motion_blur, salt_and_pepper, dead_pixel_simulation, random_occlusion,
    lens_condensation, strong_gaussian_noise, apply_seen_artifacts, apply_unseen_artifacts
)

class EvaluatorOODDatasetStaticFusion(StaticFusionThermalDataset):
    """Dataset wrapper to apply specific artifacts at a specific severity for static fusion ablation."""
    def __init__(self, image_paths, labels, img_size=224, artifact_func=None, severity=1):
        super().__init__(image_paths, labels, img_size=img_size)
        self.artifact_func = artifact_func
        self.severity = severity
        
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        img = cv2.imread(img_path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply specific artifact if provided
        if self.artifact_func is not None:
            img_gray = self.artifact_func(img_gray, severity=self.severity)
            img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR) 
            
        mask_original = self.generate_hotspot_mask(img_gray)
        img_norm = img_gray.astype(np.float32)
        img_norm = (img_norm - np.min(img_norm)) / (np.max(img_norm) + 1e-6)
        motor_mask_original = (img_norm > np.percentile(img_norm, 40)).astype(np.uint8)
        
        from dataset import extract_physics_features
        phys_feats = extract_physics_features(img_gray, mask_original, motor_mask_original)
            
        # Final tensors
        img_resized = cv2.resize(img, (self.img_size, self.img_size))
        mask_resized = cv2.resize(mask_original, (self.img_size, self.img_size))
        
        img_tensor = torch.tensor(img_resized / 255.0).permute(2, 0, 1).float()
        mask_tensor = torch.tensor(mask_resized).unsqueeze(0).float()
        phys_tensor = torch.tensor(phys_feats).float()
        
        return img_tensor, mask_tensor, phys_tensor, label


def run_evaluation_static_fusion(model, device, test_paths, test_labels, artifact_func=None, severity=1):
    dataset = EvaluatorOODDatasetStaticFusion(test_paths, test_labels, 224, artifact_func, severity)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)
    
    all_preds, all_true = [], []
    with torch.no_grad():
        for imgs, masks, phys, lbs in loader:
            imgs, masks, phys = imgs.to(device), masks.to(device), phys.to(device)
            out = model(imgs, masks, phys) 
            preds = torch.argmax(out, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_true.extend(lbs.numpy())
            
    acc = accuracy_score(all_true, all_preds)
    return acc, all_true, all_preds


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 1. Load Split
    with open("ood_split_info.json", "r") as f:
        split_info = json.load(f)
    
    test_paths = split_info["test_paths"]
    test_labels = split_info["test_labels"]
    
    print(f"Loaded FIXED untouched test split with {len(test_paths)} images")

    # 2. Load trained model
    model = StaticFusionResNet(num_classes=11).to(device)
    model.load_state_dict(torch.load("thermal_model_baseline_static_ood_trained.pth", map_location=device))
    model.eval()

    # ==========================================
    # LEVEL 1: CLEAN EVALUATION
    # ==========================================
    print("\n" + "="*60)
    print("LEVEL 1: CLEAN TEST PERFORMANCE")
    print("="*60)
    acc_clean, _, _ = run_evaluation_static_fusion(model, device, test_paths, test_labels, None, 1)
    print(f"✓ Accuracy on Clean Data: {acc_clean*100:.2f}%")

    # ==========================================
    # LEVEL 2: SEEN ARTIFACT TEST
    # ==========================================
    print("\n" + "="*60)
    print("LEVEL 2: SEEN ARTIFACT PERFORMANCE")
    print("="*60)
    
    seen_severities = range(1, 6)
    acc_seen_avg = []
    for s in seen_severities:
        acc, _, _ = run_evaluation_static_fusion(model, device, test_paths, test_labels, apply_seen_artifacts, s)
        acc_seen_avg.append(acc)
        print(f"  Severity {s}: {acc*100:.2f}%")

    # ==========================================
    # LEVEL 3: UNSEEN ARTIFACT (OOD)
    # ==========================================
    print("\n" + "="*60)
    print("LEVEL 3: OOD / UNSEEN CORRUPTIONS")
    print("="*60)

    unseen_severities = range(1, 6)
    acc_unseen_avg = []
    
    unseen_results_per_type = {}
    unseen_types = [
        ("Motion Blur", motion_blur),
        ("Salt & Pepper", salt_and_pepper),
        ("Dead Pixels", dead_pixel_simulation),
        ("Occlusion", random_occlusion),
        ("Condensation", lens_condensation),
        ("Strong Noise", strong_gaussian_noise)
    ]
    
    for s in unseen_severities:
        acc, _, _ = run_evaluation_static_fusion(model, device, test_paths, test_labels, apply_unseen_artifacts, s)
        acc_unseen_avg.append(acc)
        print(f"  Overall OOD (Random Type) - Severity {s}: {acc*100:.2f}%")

    for name, func in unseen_types:
        type_accs = []
        for s in unseen_severities:
            acc, _, _ = run_evaluation_static_fusion(model, device, test_paths, test_labels, func, s)
            type_accs.append(acc)
        unseen_results_per_type[name] = type_accs
    
    # Save JSON results
    res_dict = {
        "Clean": acc_clean,
        "Seen_Average": list(acc_seen_avg),
        "Unseen_Average": list(acc_unseen_avg),
        "Breakdown": unseen_results_per_type
    }
    with open("ood_results_summary_baseline_static.json", "w") as f:
        json.dump(res_dict, f, indent=2)

    print("\n✓ Static Fusion Metrics saved to ood_results_summary_baseline_static.json")

if __name__ == "__main__":
    main()
