# 🔥 Physics-Guided Robust Thermal Fault Diagnosis

**A Physics-Guided Convolutional Neural Network (PG-CNN) for robust motor fault classification in noisy industrial environments.**

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Robustness](https://img.shields.io/badge/Robustness-High-brightgreen)

## 📖 Overview

Standard deep learning models often fail when deployed in harsh industrial environments with sensor noise. This project implements a **Physics-Guided CNN** that integrates thermal domain knowledge (temperature statistics, hotspot morphology) to ensure reliable fault diagnosis even when image quality degrades.

### ✨ Key Research Contributions
- **� 5-Fold Cross Validation**: Achieves **100.00% ± 0.00%** accuracy across all 369 images.
- **🛡️ OOD Robustness Pipeline**: Actively tested across 3 distribution-shift levels (Clean, Seen Artifacts, Unseen OOD Corruptions).
- **🧠 Adaptive Fusion**: Automatically switches trust to physical morphology features when visual textures are destroyed by noise.
- **🎓 Curriculum Learning**: 3-stage progressive training logic (Physics Aug → Artifact Enriched → Clean).

---

## 📊 Evaluation 1: Gaussian Noise (Stress Test)

While standard models achieve 100% accuracy on clean data, our **Physics-Guided approach** is drastically more stable under simulated uniform sensor noise.

| Noise Level ($\sigma$) | Physics-Guided (Ours) | Baseline ResNet18 | **Improvement** |
| :---: | :---: | :---: | :---: |
| **0.00 (Clean)** | **100.00%** | **100.00%** | Tie |
| **0.05 (Slight)** | **100.00%** | 31.08% | **+68.92% (Massive)** |
| **0.10 (Moderate)**| **72.97%** | 25.68% | **+47.30% (Massive)** |

---

## 📊 Evaluation 2: 3-Level OOD Robustness (New!)

To truly test out-of-distribution (OOD) generalization without data leakage, we implemented a strict 70/30 split, trained the model with dynamically injected "Seen Artifacts" (Stripes, Hotspots, Drift), and tested it against completely "Unseen" corruptions.

| Evaluation Level | Condition | Accuracy | Robustness Score |
| :--- | :--- | :---: | :---: |
| **Level 1** | Untouched Clean Test Set | **100.00%** | 1.00 |
| **Level 2** | Seen Artifacts (Average Sev. 1-5) | **96.94%** | 0.97 |
| **Level 3** | Unseen OOD Corruptions | **57.12%** | 0.57 |

**OOD Breakdown Insights:**
- **Highly Robust:** Occlusion (97.8%), Motion Blur (94.2%) — Physical morphology survives.
- **Vulnerable:** Lens Condensation (47.9%), Salt/Pepper (13.3%) — Thresholds and shapes are destroyed.

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/AlenMJohn82/thermal-fault-diagnosis.git
cd thermal-fault-diagnosis

# Create environment
conda create -n thermal python=3.10 -y
conda activate thermal

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Setup
Place the base 11-class thermal datasets into:
`thermal ds-20260208T133253Z-1-001/thermal ds/`

### 3. Running Inference (Web UI)

Use the pre-trained models to classify images directly in your browser.

```bash
python app.py
```
Open **http://localhost:5000**

---

## 🧪 Reproducing Research Results

### A. 5-Fold Cross-Validation
Run the statistical validation script (trains 5 distinct models internally).
```bash
python kfold_crossval.py
```

### B. OOD Robustness Pipeline
Train the artifact-injected robust model and immediately evaluate across all 3 stress levels.
```bash
python train_ood.py   # Trains the model using curriculum + artifact injection
python eval_ood.py    # Generates the 3-level evaluation and degradation plots
```
**Output**: `ood_robustness_plot.png` and `ood_results_summary.json`

### C. Gaussian Noise Comparison
Compare the standard model vs. a plain ResNet baseline.
```bash
python noise_test.py
python generate_robustness_plot.py
```

---

## 📁 Core Project Structure

```
thermal-fault-diagnosis/
├── train_ood.py                # OOD Curriculum Training Script
├── eval_ood.py                 # 3-Level Robustness Evaluation Script
├── thermal_artifacts.py        # Seen/Unseen Artifact Generation Library
├── kfold_crossval.py           # 5-Fold Statistical Validation
├── model.py                    # PG-CNN Architecture Definition
├── dataset.py                  # Physics Feature Extraction Logic
├── app.py                      # Interactive Web Interface (Flask)
├── thermal_model_final.pth     # Standard Weights (80/20)
├── thermal_model_ood_trained.pth # OOD-Robust Weights (70/30 fixed split)
└── templates/
    └── index.html              # UI Frontend
```

## 📝 Citation

If you use this work, please cite:

```bibtex
@software{thermal_fault_diagnosis,
  author = {Alen Adon},
  title = {Physics-Guided Deep Learning for Robust Thermal Fault Diagnosis},
  year = {2026},
  url = {https://github.com/AlenMJohn82/thermal-fault-diagnosis}
}
```

## 📧 Contact
For questions or collaboration: **alenadon82@gmail.com**
