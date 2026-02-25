import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from model import PhysicsGuidedCNN
from dataset import ThermalDataset
import cv2

# Import artifact functions
from thermal_artifacts import (
    add_stripe_noise, add_gradient_drift, inject_local_hotspot, global_thermal_bias,
    motion_blur, salt_and_pepper, dead_pixel_simulation, random_occlusion,
    lens_condensation, strong_gaussian_noise, apply_seen_artifacts, apply_unseen_artifacts
)

class EvaluatorOODDataset(ThermalDataset):
    """Dataset wrapper to apply specific artifacts at a specific severity."""
    def __init__(self, image_paths, labels, img_size=224, artifact_func=None, severity=1):
        super().__init__(image_paths, labels, transform=None, img_size=img_size)
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
            img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR) # Return to 3 channels for CNN
            
        # Generate masks and extract exact features from corrupted image
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
        
        return img_tensor, mask_tensor, label, phys_tensor


def run_evaluation(model, device, test_paths, test_labels, artifact_func=None, severity=1):
    dataset = EvaluatorOODDataset(test_paths, test_labels, 224, artifact_func, severity)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)
    
    all_preds, all_true = [], []
    with torch.no_grad():
        for imgs, masks, lbs, phys in loader:
            imgs, masks, phys = imgs.to(device), masks.to(device), phys.to(device)
            out, _ = model(imgs, masks, phys)
            preds = torch.argmax(out, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_true.extend(lbs.numpy())
            
    acc = accuracy_score(all_true, all_preds)
    return acc, all_true, all_preds


def plot_cm(all_true, all_preds, class_names, title, filename):
    cm = confusion_matrix(all_true, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 1. Load Split
    with open("ood_split_info.json", "r") as f:
        split_info = json.load(f)
    
    test_paths = split_info["test_paths"]
    test_labels = split_info["test_labels"]
    class_map = {int(k): v for k, v in split_info["class_names"].items()}
    class_names = [class_map[i] for i in range(11)]
    
    print(f"Loaded FIXED untouched test split with {len(test_paths)} images")

    # 2. Load trained model
    model = PhysicsGuidedCNN(num_classes=11).to(device)
    model.load_state_dict(torch.load("thermal_model_ood_trained.pth", map_location=device))
    model.eval()

    # ==========================================
    # LEVEL 1: CLEAN EVALUATION
    # ==========================================
    print("\n" + "="*60)
    print("LEVEL 1: CLEAN TEST PERFORMANCE")
    print("="*60)
    acc_clean, true_l1, preds_l1 = run_evaluation(model, device, test_paths, test_labels, None, 1)
    print(f"✓ Accuracy on Clean Data: {acc_clean*100:.2f}%")
    plot_cm(true_l1, preds_l1, class_names, 'Confusion Matrix - Clean (Level 1)', 'cm_level1_clean.png')

    # ==========================================
    # LEVEL 2: SEEN ARTIFACT TEST
    # ==========================================
    print("\n" + "="*60)
    print("LEVEL 2: SEEN ARTIFACT PERFORMANCE (Stripe, Drift, Hotspot, Bias)")
    print("="*60)
    
    seen_severities = range(1, 6)
    acc_seen_avg = []
    preds_cm_l2 = None
    
    for s in seen_severities:
        # Evaluate using Random Seen Artifact Wrapper across the whole set
        acc, true_l2, preds_l2 = run_evaluation(model, device, test_paths, test_labels, apply_seen_artifacts, s)
        acc_seen_avg.append(acc)
        print(f"  Severity {s}: {acc*100:.2f}%")
        if s == 3: # Save CM at moderate severity
            preds_cm_l2 = preds_l2

    plot_cm(test_labels, preds_cm_l2, class_names, 'Confusion Matrix - Seen Artifacts Sev:3 (Level 2)', 'cm_level2_seen.png')
    
    # ==========================================
    # LEVEL 3: UNSEEN ARTIFACT (OOD)
    # ==========================================
    print("\n" + "="*60)
    print("LEVEL 3: OOD / UNSEEN CORRUPTIONS (Motion Blur, Condensation, etc.)")
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
    
    # Run the generic mixed OOD test
    preds_cm_l3 = None
    for s in unseen_severities:
        acc, true_l3, preds_l3 = run_evaluation(model, device, test_paths, test_labels, apply_unseen_artifacts, s)
        acc_unseen_avg.append(acc)
        print(f"  Overall OOD (Random Type) - Severity {s}: {acc*100:.2f}%")
        if s == 3:
            preds_cm_l3 = preds_l3

    plot_cm(test_labels, preds_cm_l3, class_names, 'Confusion Matrix - Unseen OOD Sev:3 (Level 3)', 'cm_level3_unseen.png')

    # Run break-down by specific unseen type to find weaknesses
    for name, func in unseen_types:
        type_accs = []
        for s in unseen_severities:
            acc, _, _ = run_evaluation(model, device, test_paths, test_labels, func, s)
            type_accs.append(acc)
        unseen_results_per_type[name] = type_accs
    
    # ==========================================
    # FINAL METRICS & PLOT
    # ==========================================
    print("\n" + "="*60)
    print("ROBUSTNESS METRICS SUMMARY")
    print("="*60)
    
    mean_acc_seen = np.mean(acc_seen_avg)
    mean_acc_ood = np.mean(acc_unseen_avg)
    
    rob_score_seen = mean_acc_seen / acc_clean
    rob_score_ood = mean_acc_ood / acc_clean
    gap_ood = acc_clean - mean_acc_ood

    print(f"Base Clean Accuracy : {acc_clean*100:.2f}%")
    print(f"Mean Seen-Artifact  : {mean_acc_seen*100:.2f}%  (Robustness Score: {rob_score_seen:.2f})")
    print(f"Mean OOD-Artifact   : {mean_acc_ood*100:.2f}%  (Robustness Score: {rob_score_ood:.2f})")
    print(f"Robustness Gap (OOD): {gap_ood*100:.2f}%\n")
    
    print("OOD Breakdown by Corruption Type (Mean Accuracy across Severities 1-5):")
    for k, v in unseen_results_per_type.items():
        print(f"  {k.ljust(15)} : {np.mean(v)*100:.2f}%")

    # Generate the comparison plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 6), [acc_clean]*5, 'k--', label='Clean Baseline', linewidth=2)
    plt.plot(range(1, 6), acc_seen_avg, 'b-o', label='Seen Artifacts Average', linewidth=2)
    plt.plot(range(1, 6), acc_unseen_avg, 'r-o', label='Unseen/OOD Average', linewidth=2)
    
    plt.title("OOD Robustness Performance Degradation", fontsize=14)
    plt.xlabel("Severity Level", fontsize=12)
    plt.ylabel("Test Accuracy", fontsize=12)
    plt.ylim([0, 1.05])
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('ood_robustness_plot.png', dpi=300)
    
    # Save JSON results
    res_dict = {
        "Clean": acc_clean,
        "Seen_Average": list(acc_seen_avg),
        "Unseen_Average": list(acc_unseen_avg),
        "Breakdown": unseen_results_per_type,
        "Robustness_Score_OOD": rob_score_ood,
        "Robustness_Gap": gap_ood
    }
    with open("ood_results_summary.json", "w") as f:
        json.dump(res_dict, f, indent=2)

    print("\n✓ Robustness plot saved to ood_robustness_plot.png")
    print("✓ Metrics saved to ood_results_summary.json")

if __name__ == "__main__":
    main()
