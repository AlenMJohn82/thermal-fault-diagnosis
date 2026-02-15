# 🔥 Physics-Guided Robust Thermal Fault Diagnosis

**A Physics-Guided Convolutional Neural Network (PG-CNN) for robust motor fault classification in noisy industrial environments.**

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Robustness](https://img.shields.io/badge/Robustness-High-brightgreen)

## 📖 Overview

Standard deep learning models often fail when deployed in harsh industrial environments with sensor noise. This project implements a **Physics-Guided CNN** that integrates thermal domain knowledge (temperature statistics, hotspot morphology) to ensuring reliable fault diagnosis even when image quality degrades.

### ✨ Key Innovations
- **🛡️ Noise Robustness**: Maintains **100% accuracy** at noise levels where standard ResNet18 drops to **31%**.
- **🧠 Adaptive Fusion**: Automatically switches trust to mechanical physics features when visual textures are noisy.
- **🎓 Curriculum Learning**: 3-stage progressive training logic (Augmented → Stochastic → Real).
- **🚀 100% Clean Accuracy**: Perfect classification on held-out test data.

---

## 📊 Key Result: Superior Robustness

While both our model and baseline methods achieve 100% accuracy on clean data, our **Physics-Guided approach** is drastically more stable under simulated sensor noise.

| Noise Level ($\sigma$) | Physics-Guided (Ours) | Baseline ResNet18 | **Improvement** |
| :---: | :---: | :---: | :---: |
| **0.00 (Clean)** | **100.00%** | **100.00%** | Tie |
| **0.05 (Slight)** | **100.00%** | 31.08% | **+68.92% (Massive)** |
| **0.10 (Moderate)**| **72.97%** | 25.68% | **+47.30% (Massive)** |

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/alenadon82/thermal-fault-diagnosis.git
cd thermal-fault-diagnosis

# Create environment
conda create -n thermal python=3.10 -y
conda activate thermal

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Setup
The dataset is too large for GitHub. Only the code and trained models are included.
To train from scratch, place your dataset in:
`thermal ds-20260208T133253Z-1-001/thermal ds/`

### 3. Running Inference (Web UI)

Use the pre-trained model (`thermal_model_final.pth`) to classify images immediately.

```bash
python app.py
```
Open **http://localhost:5000** in your browser.

---

## 🧪 Reproducing Research Results

### A. Verify Classification Performance (Clean Data)
Run the detailed evaluation script to generate the confusion matrix and classification report.

```bash
python evaluate_detailed.py
```
**Output**: `classification_metrics.txt` and `confusion_matrix.png`

### B. Verify Noise Robustness (The "Stress Test")
Run the noise experiment to see the Physics-Guided model in action against a baseline.

```bash
python noise_test.py
```
**Output**: `noise_robustness_results.json` containing accuracy at noise levels 0.0 to 0.5.

### C. Generate Paper Plot
Visualize the robustness gap between our model and the baseline.

```bash
python generate_robustness_plot.py
```
**Output**: `noise_robustness_plot.png` (Figure 3 in the paper).

---

## ⚙️ Training From Scratch

If you have the dataset, you can retrain the model using the 3-stage curriculum learning strategy.

```bash
# Train with default settings
python train.py

# Custom training
python train.py --epochs_stage1 30 --lr 0.001
```

**Note**: The training script `train.py` automatically handles **Data Leakage Prevention** by filtering augmented images that correspond to the test set.

---

## 📁 Project Structure

```
thermal-fault-diagnosis/
├── train.py                    # Main training script (Curriculum Learning)
├── model.py                    # PG-CNN Architecture Definition
├── dataset.py                  # Physics Feature Extraction Logic
├── noise_test.py               # Robustness Experiment Script
├── evaluate_detailed.py        # Classification Metrics Script
├── app.py                      # Web Interface (Flask)
├── thermal_model_final.pth     # Trained Model Weights
├── test_split_info.json        # List of Test Images (for reproducibility)
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
  url = {https://github.com/alenadon82/thermal-fault-diagnosis}
}
```

## 📧 Contact
For questions or collaboration: **alenadon82@gmail.com**
