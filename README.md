# Thermal Fault Diagnosis System

Physics-Guided Deep Learning for Motor Fault Classification using Thermal Infrared Images.

## 🎯 Features

- **3-Stage Curriculum Learning** for robust training
- **Physics-Guided Feature Fusion** with hotspot mask generation
- **11 Fault Classes** detection
- **Web UI** for easy inference and visualization
- **Test Split Tracking** - know exactly which images weren't seen during training

## 📋 Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)

## 🚀 Setup

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Verify dataset structure:**
```
thermal ds-20260208T133253Z-1-001/thermal ds/
├── Augmented_Separate_Physics_Dataset/
├── Augmented_Combined_Stochastic/
└── IR-Motor-bmp/
    ├── A10/
    ├── A30/
    ├── A50/
    ├── A&C10/
    ├── A&C30/
    ├── A&B50/
    ├── A&C&B10/
    ├── A&C&B30/
    ├── Fan/
    ├── Noload/
    └── Rotor-0/
```

## 🎓 Training

Train the model using 3-stage curriculum learning:

```bash
python train.py
```

This will:
- **Stage 1** (20 epochs): Train on separate physics augmentations
- **Stage 2** (20 epochs): Train on combined stochastic augmentations
- **Stage 3** (10 epochs): Fine-tune on clean data

### Training Outputs

- `thermal_model_final.pth` - Trained model weights
- `test_split_info.json` - **List of test images NOT seen during training**
- `checkpoint_stage1.pth` - Stage 1 checkpoint
- `checkpoint_stage2.pth` - Stage 2 checkpoint

### Custom Training Options

```bash
python train.py --epochs_stage1 30 --epochs_stage2 30 --epochs_stage3 15 --batch_size 16
```

## 🔍 Testing with Unseen Images

After training, check `test_split_info.json` to see which images were held out for testing:

```json
{
  "test_images": [
    "path/to/test/image1.bmp",
    "path/to/test/image2.bmp",
    ...
  ],
  "test_labels": [0, 1, 2, ...],
  "note": "These images were NOT seen during training"
}
```

Use these images to verify the model's performance on truly unseen data!

## 🌐 Web UI

Start the web interface:

```bash
python app.py
```

Then open your browser to: **http://localhost:5000**

### Features:
- 📤 Drag & drop image upload
- 🔥 Real-time fault classification
- 📊 Confidence scores for all classes
- 🎨 Hotspot mask visualization
- ⚡ Physics reliability indicator

## 📊 Fault Classes

| Class | Description |
|-------|-------------|
| A10 | Phase A fault - 10% severity |
| A30 | Phase A fault - 30% severity |
| A50 | Phase A fault - 50% severity |
| A&C10 | Combined Phase A & C - 10% |
| A&C30 | Combined Phase A & C - 30% |
| A&B50 | Combined Phase A & B - 50% |
| A&C&B10 | Multi-phase fault - 10% |
| A&C&B30 | Multi-phase fault - 30% |
| Fan | Fan failure |
| Rotor-0 | Rotor fault |
| Noload | No load condition |

## 🏗️ Architecture

- **Backbone**: ResNet18 (pretrained on ImageNet)
- **Physics Features**: Area ratio, ΔT, std, compactness
- **Fusion**: Physics-guided feature weighting
- **Input Size**: 224×224 RGB images

## 📁 Project Structure

```
thermal/
├── train.py              # Main training script
├── app.py                # Flask web application
├── model.py              # Model architecture
├── dataset.py            # Dataset and preprocessing
├── requirements.txt      # Python dependencies
├── templates/
│   └── index.html        # Web UI frontend
└── static/
    └── style.css         # UI styling
```

## 🎯 Expected Performance

- Test Accuracy: >90% on clean held-out data
- Training Time: ~2-3 hours on single GPU
- Inference: <1 second per image

## 🔬 Methodology

### Curriculum Learning Approach

The model is trained progressively from easy→hard:
1. **Augmented data** builds robust features
2. **Stochastic augmentations** teach generalization  
3. **Clean data** fine-tunes for real-world performance

### Physics-Guided Fusion

- Automatically generates hotspot masks
- Extracts physics features (temperature, area, shape)
- Learns reliability weighting (α) for feature fusion

## 📝 Citation

Based on the thermal fault diagnosis research with physics-guided deep learning approach.

## 🛠️ Troubleshooting

**Issue**: Model file not found
- **Solution**: Run `python train.py` first to train the model

**Issue**: CUDA out of memory
- **Solution**: Reduce batch size: `python train.py --batch_size 16`

**Issue**: Web UI not loading
- **Solution**: Check that Flask is installed and port 5000 is available

## 📧 Contact

For questions about the implementation, refer to the original notebook or implementation plan.
