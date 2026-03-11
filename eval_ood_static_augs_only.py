import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

from model import PhysicsGuidedCNN
from eval_ood_training_strategies import evaluate_model
import cv2

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    with open("ood_split_info.json", "r") as f:
        split_info = json.load(f)
    test_paths = split_info["test_paths"]
    test_labels = split_info["test_labels"]

    print("\n" + "="*60)
    print("EVALUATING STATIC-AUGS-ONLY MODEL (NO DYNAMIC OOD)")
    print("="*60)
    evaluate_model(
        "thermal_model_static_augs_only_trained.pth", 
        test_paths, test_labels, device, 
        "ood_results_summary_static_augs_only.json"
    )

if __name__ == "__main__":
    main()
