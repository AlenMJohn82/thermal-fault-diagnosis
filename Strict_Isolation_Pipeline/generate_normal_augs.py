import os
import cv2
import numpy as np
import random
import glob

def rotate(img):
    angle = random.choice([90, 180, 270, 15, -15, 30, -30, 45, -45])
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    return cv2.warpAffine(img, M, (w, h))

def flip(img):
    return cv2.flip(img, random.choice([0, 1, -1]))

def crop(img):
    h, w = img.shape[:2]
    crop_scale = random.uniform(0.6, 0.9)
    ch, cw = int(h*crop_scale), int(w*crop_scale)
    y = random.randint(0, h - ch)
    x = random.randint(0, w - cw)
    cropped = img[y:y+ch, x:x+cw]
    return cv2.resize(cropped, (w, h))

def scale(img):
    h, w = img.shape[:2]
    scale_factor = random.uniform(0.5, 1.5)
    resized = cv2.resize(img, None, fx=scale_factor, fy=scale_factor)
    if scale_factor > 1.0:
        nh, nw = resized.shape[:2]
        y = (nh - h) // 2
        x = (nw - w) // 2
        return resized[y:y+h, x:x+w]
    else:
        nh, nw = resized.shape[:2]
        pad_y = (h - nh) // 2
        pad_x = (w - nw) // 2
        padded = np.zeros((h, w, 3 if len(img.shape)==3 else 1), dtype=img.dtype)
        padded[pad_y:pad_y+nh, pad_x:pad_x+nw] = resized
        return padded

def generate_normal_augmentations():
    in_dir = "Strict_OOD_Pipeline/TRAIN/Clean"
    out_sep = "Strict_OOD_Pipeline/TRAIN/Normal_Aug_Separate"
    out_comb = "Strict_OOD_Pipeline/TRAIN/Normal_Aug_Combined"

    print("="*60)
    print("GENERATING STANDARD GEOMETRIC AUGMENTATIONS")
    print("="*60)

    classes = os.listdir(in_dir)
    for c in classes:
        os.makedirs(os.path.join(out_sep, c), exist_ok=True)
        os.makedirs(os.path.join(out_comb, c), exist_ok=True)

    total_images = glob.glob(os.path.join(in_dir, "*", "*.bmp"))
    print(f"Found {len(total_images)} original training images.")

    sep_count = 0
    comb_count = 0
    
    for img_path in total_images:
        img = cv2.imread(img_path)
        cls_name = os.path.basename(os.path.dirname(img_path))
        base_name = os.path.splitext(os.path.basename(img_path))[0]

        # Stage 1: Separate (10 variations total per image to match 10x dataset size)
        # 3 Rotations, 2 Flips, 3 Crops, 2 Scales
        sep_funcs = [rotate, rotate, rotate, flip, flip, crop, crop, crop, scale, scale]
        for i, func in enumerate(sep_funcs):
            aug_img = func(img)
            out_path = os.path.join(out_sep, cls_name, f"{base_name}_sep_{i}.bmp")
            cv2.imwrite(out_path, aug_img)
            sep_count += 1

        # Stage 2: Combined (10 variations total per image)
        # 3x (rot+flip), 3x (crop+scale), 4x (rot+crop+scale)
        for i in range(10):
            aug_img = img.copy()
            if i < 3:
                aug_img = flip(rotate(aug_img))
            elif i < 6:
                aug_img = scale(crop(aug_img))
            else:
                aug_img = scale(crop(rotate(aug_img)))
            
            out_path = os.path.join(out_comb, cls_name, f"{base_name}_comb_{i}.bmp")
            cv2.imwrite(out_path, aug_img)
            comb_count += 1

    print(f"Generated {sep_count} Separate Geometric augmentations.")
    print(f"Generated {comb_count} Combined Geometric augmentations.")
    print("Done!")

if __name__ == "__main__":
    generate_normal_augmentations()
