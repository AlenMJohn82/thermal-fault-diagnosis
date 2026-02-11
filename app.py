import os
import json
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import base64

from model import PhysicsGuidedCNN
from dataset import extract_physics_features


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Class mapping
CLASS_MAP = {
    "A10": 0, "A30": 1, "A50": 2, "A&C10": 3, "A&C30": 4,
    "A&B50": 5, "A&C&B10": 6, "A&C&B30": 7,
    "Fan": 8, "Rotor-0": 9, "Noload": 10
}
CLASS_NAMES = {v: k for k, v in CLASS_MAP.items()}

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = PhysicsGuidedCNN(num_classes=11).to(device)

# Load trained weights
model_path = "thermal_model_final.pth"
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"✓ Model loaded from {model_path}")
else:
    print(f"⚠ Warning: Model file not found at {model_path}")
    print("Please train the model first using: python train.py")


def generate_hotspot_mask(img_gray):
    """Generate hotspot mask from thermal image"""
    img = img_gray.astype(np.float32)
    img = img - np.min(img)
    img = img / (np.max(img) + 1e-6)
    
    motor_mask = img > np.percentile(img, 40)
    img_motor = img * motor_mask
    
    mean = np.mean(img_motor[img_motor > 0])
    std = np.std(img_motor[img_motor > 0])
    
    hotspot = img_motor > (mean + 0.8 * std)
    
    if np.sum(hotspot) < 100:
        hotspot = img_motor > (mean + 0.5 * std)
    
    hotspot = hotspot.astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    hotspot = cv2.morphologyEx(hotspot, cv2.MORPH_OPEN, kernel)
    hotspot = cv2.morphologyEx(hotspot, cv2.MORPH_CLOSE, kernel)
    
    hotspot = hotspot & motor_mask.astype(np.uint8)
    
    if np.sum(hotspot) < 50:
        hotspot = motor_mask.astype(np.uint8)
    
    return hotspot, motor_mask.astype(np.uint8)


def process_image(image_path):
    """Process image and return prediction"""
    # Load image
    img = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Generate hotspot mask
    mask_original, motor_mask_original = generate_hotspot_mask(img_gray)
    
    # Extract physics features
    phys_feats = extract_physics_features(img_gray, mask_original, motor_mask_original)
    
    # Resize for model
    img_resized = cv2.resize(img, (224, 224))
    mask_resized = cv2.resize(mask_original, (224, 224))
    
    # Convert to tensors
    img_tensor = torch.tensor(img_resized / 255.0).permute(2, 0, 1).float().unsqueeze(0)
    mask_tensor = torch.tensor(mask_resized).unsqueeze(0).unsqueeze(0).float()
    phys_tensor = torch.tensor(phys_feats).unsqueeze(0).float()
    
    # Predict
    with torch.no_grad():
        img_tensor = img_tensor.to(device)
        mask_tensor = mask_tensor.to(device)
        phys_tensor = phys_tensor.to(device)
        
        outputs, alpha = model(img_tensor, mask_tensor, phys_tensor)
        probs = F.softmax(outputs, dim=1)
        pred_class = torch.argmax(outputs, dim=1).item()
        confidence = probs[0, pred_class].item()
    
    # Create visualization - overlay hotspot on original image
    img_viz = img.copy()
    overlay = cv2.applyColorMap(mask_original * 255, cv2.COLORMAP_JET)
    img_viz = cv2.addWeighted(img_viz, 0.6, overlay, 0.4, 0)
    
    # Convert to base64 for sending to frontend
    _, buffer = cv2.imencode('.png', img_viz)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return {
        "predicted_class": CLASS_NAMES[pred_class],
        "confidence": float(confidence),
        "all_probabilities": {CLASS_NAMES[i]: float(probs[0, i].item()) for i in range(11)},
        "visualization": img_base64,
        "physics_reliability": float(alpha.item())
    }


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            result = process_image(filepath)
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        finally:
            # Clean up uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)


if __name__ == '__main__':
    print("\n" + "="*50)
    print("THERMAL FAULT DIAGNOSIS WEB UI")
    print("="*50)
    print(f"Device: {device}")
    print(f"Model: {model_path}")
    print("\nStarting server at http://localhost:5000")
    print("="*50 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
