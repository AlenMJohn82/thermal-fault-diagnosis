# 🔥 Thermal Fault Diagnosis System

Physics-Guided Deep Learning for Motor Fault Classification using Thermal Infrared Images.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 📖 Overview

This project implements a **physics-guided CNN** for detecting motor faults from thermal infrared images. It combines deep learning with domain knowledge through:

- ✨ **Physics-guided feature fusion** - Integrates visual patterns with thermal physics
- 🎓 **Curriculum learning** - Progressive training from augmented to real data
- 🎯 **100% test accuracy** achieved on held-out images
- 🌐 **Interactive web UI** for easy inference and visualization

## 🚀 Quick Start (For New Developers)

### Prerequisites

Make sure you have:
- Python 3.8 or higher installed
- (Optional) NVIDIA GPU with CUDA for faster training
- Git installed

### 1. Clone the Repository

```bash
git clone https://github.com/alenadon82/thermal-fault-diagnosis.git
cd thermal-fault-diagnosis
```

### 2. Set Up Environment

**Option A: Using Conda (Recommended)**

```bash
# Create conda environment
conda create -n thermal python=3.10 -y
conda activate thermal

# Install dependencies
pip install torch torchvision opencv-python numpy scikit-learn Flask
```

**Option B: Using pip + venv**

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Download the Dataset

⚠️ **Important**: The dataset is NOT included in this repository (too large for GitHub).

You need to download the thermal images dataset separately and place it in this structure:

```
thermal-fault-diagnosis/
├── thermal ds-20260208T133253Z-1-001/
│   └── thermal ds/
│       ├── Augmented_Separate_Physics_Dataset/
│       ├── Augmented_Combined_Stochastic/
│       └── IR-Motor-bmp/
│           ├── A10/
│           ├── A30/
│           └── ... (11 fault class folders)
└── (other project files)
```

Contact the project owner for dataset access.

### 4. Use Pre-trained Model OR Train Your Own

**Option A: Use Pre-trained Model** (Included in repo)

The repository includes trained model weights:
- `thermal_model_final.pth` - Ready to use for inference
- Skip to step 5 to run the web UI!

**Option B: Train From Scratch**

If you have the dataset, train your own model:

```bash
python train.py
```

Training takes ~30-40 minutes on GPU (2-3 hours on CPU). You'll see:

```
Stage 1: SEPARATE PHYSICS AUGMENTATIONS
Epoch 1/20 | Loss: 0.3842
...
Stage 2: COMBINED STOCHASTIC AUGMENTATIONS
...
Stage 3: CLEAN DATA FINE-TUNING
...
✓ Final model saved to: thermal_model_final.pth
Test Accuracy: 100.00%
```

### 5. Run the Web UI

```bash
python app.py
```

Open your browser to: **http://localhost:5000**

You'll see a beautiful purple gradient interface where you can:
- 📤 Upload thermal images (drag & drop)
- 🔥 Get instant fault predictions
- 📊 See confidence scores
- 🎨 View hotspot visualizations

### 6. Test with Sample Images

If you have `test_split_info.json`, it contains paths to test images:

```bash
# View test images that weren't seen during training
cat test_split_info.json
```

Upload any of these images to the web UI to verify the model!

## 📚 How It Works

### The 11 Fault Classes

The system can detect these motor conditions:

| Class | Description | Thermal Pattern |
|-------|-------------|-----------------|
| **A10** | Phase A fault - 10% severity | Small hotspot in phase A winding |
| **A30** | Phase A fault - 30% severity | Medium hotspot in phase A |
| **A50** | Phase A fault - 50% severity | Large hotspot in phase A |
| **A&C10** | Phase A & C fault - 10% | Two small hotspots |
| **A&C30** | Phase A & C fault - 30% | Two medium hotspots |
| **A&B50** | Phase A & B fault - 50% | Two large hotspots |
| **A&C&B10** | All phases - 10% | Multiple small hotspots |
| **A&C&B30** | All phases - 30% | Multiple medium hotspots |
| **Fan** | Fan failure | Overall heat buildup |
| **Rotor-0** | Rotor fault | Rotating hotspot pattern |
| **Noload** | No load | Uniform low temperature |

### Architecture

```
Input Image (224×224)
    ↓
ResNet18 Backbone ──────────┐
    ↓                        │
Visual Features          Physics Features
    ↓                    (area, ΔT, std)
    ↓                        ↓
    ↓                  Reliability Net
    ↓                        ↓
    └──→ Fusion (α-weighted) ──→ Classifier → Prediction
```

**Key Innovation**: The model learns how much to trust physics vs. visual features!

### Training Process (Curriculum Learning)

```
Stage 1 (20 epochs)          Stage 2 (20 epochs)          Stage 3 (10 epochs)
Augmented Physics Data   →   Stochastic Augmented    →    Clean Real Data
(easier patterns)            (moderate difficulty)        (real-world)
```

## 🎯 Performance

- **Test Accuracy**: 100% (74 held-out images)
- **Inference Speed**: < 1 second per image
- **Model Size**: 43 MB
- **All 11 classes**: Perfect precision, recall, F1-score

## 📁 Project Structure

```
thermal-fault-diagnosis/
├── train.py                    # Training script with curriculum learning
├── app.py                      # Flask web server for inference
├── model.py                    # PhysicsGuidedCNN architecture
├── dataset.py                  # Data loading & preprocessing
├── requirements.txt            # Python dependencies
├── thermal_model_final.pth     # Trained model weights (43 MB)
├── test_split_info.json        # Test images list
├── templates/
│   └── index.html              # Web UI frontend
└── static/
    └── style.css               # UI styling
```

## 🛠️ Advanced Usage

### Custom Training Parameters

```bash
# Adjust epochs and batch size
python train.py --epochs_stage1 30 --epochs_stage2 30 --epochs_stage3 15 --batch_size 16

# Use CPU only (no GPU)
python train.py --device cpu
```

### Using the Model Programmatically

```python
import torch
from model import PhysicsGuidedCNN
from dataset import ThermalDataset

# Load model
model = PhysicsGuidedCNN(num_classes=11)
model.load_state_dict(torch.load('thermal_model_final.pth'))
model.eval()

# Run inference
# (See app.py for complete example)
```

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'torch'` | Install dependencies: `pip install -r requirements.txt` |
| `FileNotFoundError: thermal_model_final.pth` | Run `python train.py` to train the model first |
| `CUDA out of memory` | Reduce batch size: `python train.py --batch_size 8` |
| Web UI shows "Model not loaded" | Make sure `thermal_model_final.pth` is in the project directory |
| Port 5000 already in use | Change port in app.py: `app.run(port=5001)` |

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@software{thermal_fault_diagnosis,
  author = {Alen Adon},
  title = {Physics-Guided Thermal Fault Diagnosis System},
  year = {2026},
  url = {https://github.com/alenadon82/thermal-fault-diagnosis}
}
```

## 📧 Contact

For questions or issues:
- Open an issue on GitHub
- Email: alenadon82@gmail.com

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- ResNet18 architecture from torchvision
- Physics-guided fusion approach based on thermal diagnostics research
- Dataset from motor fault thermal imaging experiments
