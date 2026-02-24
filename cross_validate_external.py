"""
Zero-shot cross-validation: Evaluate our trained model on ALL images from the
external Kaggle dataset (amirberenji/thermal-images-of-induction-motor).
No training — pure inference only.
"""
import os
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report
import json

from model import PhysicsGuidedCNN
from dataset import ThermalDataset, load_dataset_paths

CLASS_MAP = {
    "A10": 0, "A30": 1, "A50": 2,
    "A&C10": 3, "A&C30": 4, "A&B50": 5,
    "A&C&B10": 6, "A&C&B30": 7,
    "Fan": 8, "Rotor-0": 9, "Noload": 10
}
CLASS_NAMES = {v: k for k, v in CLASS_MAP.items()}

def evaluate_on_external(model_path, dataset_path, model_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = PhysicsGuidedCNN(num_classes=11).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    paths, labels = load_dataset_paths(dataset_path, CLASS_MAP)
    print(f"  Total external images found: {len(paths)}")

    dataset = ThermalDataset(paths, labels, img_size=224)
    loader  = DataLoader(dataset, batch_size=32, shuffle=False)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, masks, lbs, phys in loader:
            imgs, masks, phys = imgs.to(device), masks.to(device), phys.to(device)
            out, _ = model(imgs, masks, phys)
            preds = torch.argmax(out, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(lbs.numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"\n{'='*60}")
    print(f"MODEL: {model_name}")
    print(f"EXTERNAL TEST ACCURACY: {acc*100:.2f}%  ({sum(p==l for p,l in zip(all_preds,all_labels))}/{len(all_labels)} correct)")
    print(f"{'='*60}")
    print(classification_report(all_labels, all_preds,
                                target_names=[CLASS_NAMES[i] for i in range(11)],
                                digits=4))
    return acc

if __name__ == "__main__":
    ext_path = "external_dataset"

    print("\n" + "="*60)
    print("ZERO-SHOT CROSS-VALIDATION ON EXTERNAL KAGGLE DATASET")
    print("(No retraining — pure inference on unseen images)")
    print("="*60)

    # Check what's available
    print(f"\nExternal dataset path: {ext_path}")

    acc1 = evaluate_on_external(
        "thermal_model_final.pth",
        ext_path,
        "Original Model (Stochastic Stage 2, 80:20 split)"
    )

    acc2 = evaluate_on_external(
        "thermal_model_artifact.pth",
        ext_path,
        "Artifact Model (Artifact Stage 2, 70:30 split)"
    )

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"  Original Model Accuracy : {acc1*100:.2f}%")
    print(f"  Artifact Model Accuracy : {acc2*100:.2f}%")
    print(f"  Difference              : {(acc2-acc1)*100:+.2f}%")
