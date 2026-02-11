import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def extract_physics_features(img_gray, mask, motor_mask):
    """Extract physics-based features from thermal image"""
    eps = 1e-6
    
    # Area ratio
    area_ratio = mask.sum() / (motor_mask.sum() + eps)
    
    # Temperature statistics
    T_hot = img_gray[mask == 1].mean()
    T_motor = img_gray[motor_mask == 1].mean()
    delta_T = T_hot - T_motor
    std_T = img_gray[mask == 1].std()
    
    # Shape compactness
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        compactness = (perimeter ** 2) / (4 * np.pi * area + eps)
    else:
        compactness = 0.0
    
    return np.array([area_ratio, delta_T, std_T, compactness], dtype=np.float32)


class ThermalDataset(Dataset):
    """Thermal image dataset with hotspot mask generation"""
    
    def __init__(self, image_paths, labels, transform=None, img_size=224):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.img_size = img_size
    
    def generate_hotspot_mask(self, img_gray):
        """Generate hotspot mask from thermal image"""
        # Normalize
        img = img_gray.astype(np.float32)
        img = img - np.min(img)
        img = img / (np.max(img) + 1e-6)
        
        # Motor region constraint
        motor_mask = img > np.percentile(img, 40)
        img_motor = img * motor_mask
        
        # Local contrast threshold
        mean = np.mean(img_motor[img_motor > 0])
        std = np.std(img_motor[img_motor > 0])
        
        hotspot = img_motor > (mean + 0.8 * std)
        
        # Relax threshold if needed
        if np.sum(hotspot) < 100:
            hotspot = img_motor > (mean + 0.5 * std)
        
        # Morphological cleanup
        hotspot = hotspot.astype(np.uint8)
        kernel = np.ones((5, 5), np.uint8)
        hotspot = cv2.morphologyEx(hotspot, cv2.MORPH_OPEN, kernel)
        hotspot = cv2.morphologyEx(hotspot, cv2.MORPH_CLOSE, kernel)
        
        # Enforce motor boundary
        hotspot = hotspot & motor_mask.astype(np.uint8)
        
        # Safety fallback
        if np.sum(hotspot) < 50:
            hotspot = motor_mask.astype(np.uint8)
        
        return hotspot
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        img = cv2.imread(img_path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Generate hotspot mask at original size
        mask_original = self.generate_hotspot_mask(img_gray)
        
        # Generate motor mask for physics features
        img_norm = img_gray.astype(np.float32)
        img_norm = (img_norm - np.min(img_norm)) / (np.max(img_norm) + 1e-6)
        motor_mask_original = (img_norm > np.percentile(img_norm, 40)).astype(np.uint8)
        
        # Extract physics features
        phys_feats = extract_physics_features(img_gray, mask_original, motor_mask_original)
        
        # Resize for CNN
        img_resized = cv2.resize(img, (self.img_size, self.img_size))
        mask_resized = cv2.resize(mask_original, (self.img_size, self.img_size))
        
        # Convert to tensors
        img_tensor = torch.tensor(img_resized / 255.0).permute(2, 0, 1).float()
        mask_tensor = torch.tensor(mask_resized).unsqueeze(0).float()
        phys_tensor = torch.tensor(phys_feats).float()
        
        if self.transform:
            img_tensor = self.transform(img_tensor)
        
        return img_tensor, mask_tensor, label, phys_tensor
    
    def __len__(self):
        return len(self.image_paths)


def load_dataset_paths(root_dir, class_map):
    """Load all image paths and labels from dataset directory"""
    all_paths = []
    all_labels = []
    
    for class_name in sorted(os.listdir(root_dir)):
        class_path = os.path.join(root_dir, class_name)
        
        if os.path.isdir(class_path) and class_name in class_map:
            for img_name in os.listdir(class_path):
                if img_name.lower().endswith(('.bmp', '.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_path, img_name)
                    all_paths.append(img_path)
                    all_labels.append(class_map[class_name])
    
    return all_paths, all_labels
