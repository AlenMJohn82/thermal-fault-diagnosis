import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

from model import PhysicsGuidedCNN
from eval_ood import EvaluatorOODDataset, run_evaluation
import cv2
from thermal_artifacts import (
    add_stripe_noise, add_gradient_drift, inject_local_hotspot, global_thermal_bias,
    motion_blur, salt_and_pepper, dead_pixel_simulation, random_occlusion,
    lens_condensation, strong_gaussian_noise, apply_seen_artifacts, apply_unseen_artifacts
)

def evaluate_model(model_path, test_paths, test_labels, device, output_json):
    model = PhysicsGuidedCNN(num_classes=11).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # LEVEL 1
    acc_clean, _, _ = run_evaluation(model, device, test_paths, test_labels, None, 1)

    # LEVEL 2
    seen_severities = range(1, 6)
    acc_seen_avg = []
    for s in seen_severities:
        acc, _, _ = run_evaluation(model, device, test_paths, test_labels, apply_seen_artifacts, s)
        acc_seen_avg.append(acc)

    # LEVEL 3
    unseen_severities = range(1, 6)
    acc_unseen_avg = []
    for s in unseen_severities:
        acc, _, _ = run_evaluation(model, device, test_paths, test_labels, apply_unseen_artifacts, s)
        acc_unseen_avg.append(acc)

    # Save
    res_dict = {
        "Clean": acc_clean,
        "Seen_Average": list(acc_seen_avg),
        "Unseen_Average": list(acc_unseen_avg)
    }
    with open(output_json, "w") as f:
        json.dump(res_dict, f, indent=2)
    print(f"Saved metrics to {output_json}")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    with open("ood_split_info.json", "r") as f:
        split_info = json.load(f)
    test_paths = split_info["test_paths"]
    test_labels = split_info["test_labels"]

    print("\n" + "="*60)
    print("EVALUATING NO-CURRICULUM MODEL")
    print("="*60)
    evaluate_model(
        "thermal_model_no_curr_ood_trained.pth", 
        test_paths, test_labels, device, 
        "ood_results_summary_no_curr.json"
    )

    print("\n" + "="*60)
    print("EVALUATING DIRECT-ARTIFACTS MODEL")
    print("="*60)
    evaluate_model(
        "thermal_model_direct_art_ood_trained.pth", 
        test_paths, test_labels, device, 
        "ood_results_summary_direct_art.json"
    )

if __name__ == "__main__":
    main()
