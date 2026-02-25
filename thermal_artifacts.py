import cv2
import numpy as np
import random

def clamp(img):
    return np.clip(img, 0, 255).astype(np.uint8)

# ==========================================
# SEEN ARTIFACTS (In-Distribution)
# ==========================================

def add_stripe_noise(img, severity=1):
    """Adds horizontal and vertical stripe noise."""
    h, w = img.shape[:2]
    noisy = img.copy().astype(np.float32)
    num_stripes = severity * 2
    intensity = severity * 10
    
    for _ in range(num_stripes):
        if random.random() > 0.5: # Horizontal
            y = random.randint(0, h-1)
            noisy[y, :] += intensity
        else: # Vertical
            x = random.randint(0, w-1)
            noisy[:, x] += intensity
            
    return clamp(noisy)

def add_gradient_drift(img, severity=1):
    """Adds a smooth thermal drift gradient across the image."""
    h, w = img.shape[:2]
    noisy = img.copy().astype(np.float32)
    max_drift = severity * 15
    
    # Create 2D gradient
    x = np.linspace(0, 1, w)
    y = np.linspace(0, 1, h)
    xx, yy = np.meshgrid(x, y)
    
    # Random direction and weights
    angle = random.uniform(0, 2*np.pi)
    gradient = max_drift * (np.cos(angle)*xx + np.sin(angle)*yy)
    if len(noisy.shape) == 3:
        gradient = np.expand_dims(gradient, 2)
        
    noisy += gradient
    return clamp(noisy)

def inject_local_hotspot(img, severity=1):
    """Injects a fake local hotspot (2D Gaussian blob)."""
    h, w = img.shape[:2]
    noisy = img.copy().astype(np.float32)
    
    peak_intensity = severity * 25
    radius = severity * 5
    
    cx = random.randint(radius, w - radius)
    cy = random.randint(radius, h - radius)
    
    y, x = np.ogrid[-cy:h-cy, -cx:w-cx]
    g = np.exp(-(x*x + y*y) / (2.*radius*radius))
    
    blob = g * peak_intensity
    if len(noisy.shape) == 3:
        blob = np.expand_dims(blob, 2)
        
    noisy += blob
    return clamp(noisy)

def global_thermal_bias(img, severity=1):
    """Adds a global temperature bias offset."""
    noisy = img.copy().astype(np.float32)
    bias = severity * 12 # Can be up to 60 intensity units
    
    noisy += bias
    return clamp(noisy)

# ==========================================
# UNSEEN ARTIFACTS (OOD / Stress Test)
# ==========================================

def motion_blur(img, severity=1):
    """Simulates camera motion blur."""
    size = severity * 3 + 1 # 4, 7, 10, 13, 16
    kernel_motion_blur = np.zeros((size, size))
    kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
    kernel_motion_blur = kernel_motion_blur / size
    
    # Random angle
    angle = random.uniform(0, 360)
    matrix = cv2.getRotationMatrix2D((size/2, size/2), angle, 1)
    kernel_motion_blur = cv2.warpAffine(kernel_motion_blur, matrix, (size, size))
    
    return cv2.filter2D(img, -1, kernel_motion_blur)

def salt_and_pepper(img, severity=1):
    """Adds salt and pepper noise (extreme pixel failures)."""
    h, w = img.shape[:2]
    noisy = img.copy()
    prob = severity * 0.02 # 2% to 10%
    
    # Generate random matrix
    rnd = np.random.rand(h, w)
    noisy[rnd < prob/2] = 0
    noisy[rnd > 1 - prob/2] = 255
    return noisy

def dead_pixel_simulation(img, severity=1):
    """Simulates clusters of dead/stuck sensor pixels."""
    h, w = img.shape[:2]
    noisy = img.copy()
    num_clusters = severity * 5
    
    for _ in range(num_clusters):
        cx = random.randint(0, w-1)
        cy = random.randint(0, h-1)
        c_size = random.randint(1, 3) # small blocks 1x1 to 3x3
        
        y1, y2 = max(0, cy-c_size), min(h, cy+c_size)
        x1, x2 = max(0, cx-c_size), min(w, cx+c_size)
        
        # Dead is highly cold (0) or hot (255)
        val = 0 if random.random() < 0.5 else 255
        noisy[y1:y2, x1:x2] = val
        
    return noisy

def random_occlusion(img, severity=1):
    """Simulates physical occlusion (dirt on lens, object in front)."""
    h, w = img.shape[:2]
    noisy = img.copy()
    
    occ_w = int(w * (severity * 0.05 + 0.05)) # 10% to 30% width
    occ_h = int(h * (severity * 0.05 + 0.05))
    
    x = random.randint(0, w - occ_w)
    y = random.randint(0, h - occ_h)
    
    val = random.randint(0, 50) # Dark gray/black
    noisy[y:y+occ_h, x:x+occ_w] = val
    return noisy

def lens_condensation(img, severity=1):
    """Simulates lens fog/condensation (blur + vignette)."""
    noisy = img.copy().astype(np.float32)
    h, w = img.shape[:2]
    
    # 1. Blur
    k_size = severity * 4 + 1
    noisy = cv2.GaussianBlur(noisy, (k_size, k_size), 0)
    
    # 2. Vignette (darken edges)
    X_resultant_kernel = cv2.getGaussianKernel(w, w/(severity*1.5))
    Y_resultant_kernel = cv2.getGaussianKernel(h, h/(severity*1.5))
    kernel = Y_resultant_kernel * X_resultant_kernel.T
    mask = kernel / kernel.max()
    
    if len(noisy.shape) == 3:
        mask = np.expand_dims(mask, 2)
        
    noisy = noisy * mask
    return clamp(noisy)

def strong_gaussian_noise(img, severity=1):
    """High variance white noise simulating poor sensor conditions."""
    noisy = img.copy().astype(np.float32)
    std = severity * 15 # 15 to 75
    
    noise = np.random.normal(0, std, img.shape)
    noisy += noise
    return clamp(noisy)


# ==========================================
# WRAPPER TO APPLY RANDOM ARTIFACTS
# ==========================================

def apply_seen_artifacts(img, severity=1):
    """Apply 1 random seen artifact."""
    artifacts = [add_stripe_noise, add_gradient_drift, inject_local_hotspot, global_thermal_bias]
    func = random.choice(artifacts)
    return func(img, severity)

def apply_unseen_artifacts(img, severity=1):
    """Apply 1 random unseen artifact."""
    artifacts = [motion_blur, salt_and_pepper, dead_pixel_simulation, 
                 random_occlusion, lens_condensation, strong_gaussian_noise]
    func = random.choice(artifacts)
    return func(img, severity)

