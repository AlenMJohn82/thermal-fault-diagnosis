import os
import json
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

from model import PhysicsGuidedCNN
from eval_ood_training_strategies import evaluate_model
from dataset import load_dataset_paths

CLASS_MAP = {
    "A10": 0, "A30": 1, "A50": 2,
    "A&C10": 3, "A&C30": 4, "A&B50": 5,
    "A&C&B10": 6, "A&C&B30": 7,
    "Fan": 8, "Rotor-0": 9, "Noload": 10
}

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load from the Strict pipeline testing dataset
    test_paths, test_labels = load_dataset_paths("Strict_OOD_Pipeline/TEST/Clean", CLASS_MAP)

    print("\n" + "="*60)
    print(f"EVALUATING NORMAL AUGMENTATIONS MODEL (TEST SET: {len(test_paths)})")
    print("="*60)
    
    evaluate_model(
        "thermal_model_normal_augs.pth", 
        test_paths, test_labels, device, 
        "ood_results_summary_normal_augs.json"
    )
    print("\n✓ Normal Augs Pipeline Evaluation Complete!")

if __name__ == "__main__":
    main()
