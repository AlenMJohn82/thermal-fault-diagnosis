import os
import shutil
import json
from sklearn.model_selection import train_test_split
from dataset import load_dataset_paths

CLASS_MAP = {
    "A10": 0, "A30": 1, "A50": 2,
    "A&C10": 3, "A&C30": 4, "A&B50": 5,
    "A&C&B10": 6, "A&C&B30": 7,
    "Fan": 8, "Rotor-0": 9, "Noload": 10
}

def create_directories():
    base_out = "Strict_OOD_Pipeline"
    dirs = [
        f"{base_out}/TRAIN/Clean",
        f"{base_out}/TRAIN/Physics_Aug",
        f"{base_out}/TRAIN/Stoch_Aug",
        f"{base_out}/TEST/Clean"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    return base_out

def main():
    print("="*60)
    print("STEP 1: CREATING STRICT PHYSICAL DATA SPLIT (80/20)")
    print("="*60)
    
    base_in = "thermal ds-20260208T133253Z-1-001/thermal ds"
    path_clean = os.path.join(base_in, "IR-Motor-bmp")
    path_sep = os.path.join(base_in, "Augmented_Separate_Physics_Dataset")
    path_stoch = os.path.join(base_in, "Augmented_Combined_Stochastic")
    
    out_dir = create_directories()
    
    # 1. Load Original 369 Clean Images
    clean_paths, clean_labels = load_dataset_paths(path_clean, CLASS_MAP)
    
    # 2. Split 80/20
    train_p, test_p, train_l, test_l = train_test_split(
        clean_paths, clean_labels, test_size=0.20, stratify=clean_labels, random_state=42
    )
    
    print(f"Total Original Images: {len(clean_paths)}")
    print(f"Split -> TRAIN: {len(train_p)} images (80%) | TEST: {len(test_p)} images (20%)")
    
    train_basenames = {os.path.basename(p) for p in train_p}
    test_basenames = {os.path.basename(p) for p in test_p}
    
    # 3. Copy Clean TRAIN to TRAIN/Clean
    print("\nCopying Clean TRAIN images...")
    for p in train_p:
        cls_dir = os.path.basename(os.path.dirname(p))
        out_cls_dir = os.path.join(out_dir, "TRAIN/Clean", cls_dir)
        os.makedirs(out_cls_dir, exist_ok=True)
        shutil.copy(p, os.path.join(out_cls_dir, os.path.basename(p)))
        
    # 4. Copy Clean TEST to TEST/Clean
    print("Copying Clean TEST images... (NEVER TO BE TOUCHED)")
    for p in test_p:
        cls_dir = os.path.basename(os.path.dirname(p))
        out_cls_dir = os.path.join(out_dir, "TEST/Clean", cls_dir)
        os.makedirs(out_cls_dir, exist_ok=True)
        shutil.copy(p, os.path.join(out_cls_dir, os.path.basename(p)))
        
    # 5. Populate TRAIN/Physics_Aug and TRAIN/Stoch_Aug by filtering the massive datasets
    print("\nFiltering and copying Physics Augmentations (TRAIN ONLY)...")
    sep_paths_all, _ = load_dataset_paths(path_sep, CLASS_MAP)
    phys_count = 0
    for p in sep_paths_all:
        basename = os.path.basename(p)
        parts = basename.split('_')
        if basename.startswith('sep_') and len(parts) >= 3:
            original_basename = parts[1] + '.bmp'
        else:
            original_basename = basename
            
        if original_basename in train_basenames:
            cls_dir = os.path.basename(os.path.dirname(p))
            out_cls_dir = os.path.join(out_dir, "TRAIN/Physics_Aug", cls_dir)
            os.makedirs(out_cls_dir, exist_ok=True)
            shutil.copy(p, os.path.join(out_cls_dir, basename))
            phys_count += 1
    print(f"  -> Copied {phys_count} physics augmented images (10x of Train).")
    
    print("Filtering and copying Stochastic Augmentations (TRAIN ONLY)...")
    stoch_paths_all, _ = load_dataset_paths(path_stoch, CLASS_MAP)
    stoch_count = 0
    for p in stoch_paths_all:
        basename = os.path.basename(p)
        parts = basename.split('_')
        if 'stoch' in basename and len(parts) >= 3:
            original_basename = parts[0] + '.bmp'
        else:
            original_basename = basename
            
        if original_basename in train_basenames:
            cls_dir = os.path.basename(os.path.dirname(p))
            out_cls_dir = os.path.join(out_dir, "TRAIN/Stoch_Aug", cls_dir)
            os.makedirs(out_cls_dir, exist_ok=True)
            shutil.copy(p, os.path.join(out_cls_dir, basename))
            stoch_count += 1
    print(f"  -> Copied {stoch_count} stochastic augmented images (10x of Train).")
    
    # Save the strict split registry for easy access in training script
    split_registry = {
        "train_clean": [os.path.join(out_dir, "TRAIN/Clean", os.path.basename(p)) for p in train_p],
        "train_labels": [int(l) for l in train_l],
        "test_clean": [os.path.join(out_dir, "TEST/Clean", os.path.basename(p)) for p in test_p],
        "test_labels": [int(l) for l in test_l]
    }
    with open(os.path.join(out_dir, "strict_split_registry.json"), "w") as f:
        json.dump(split_registry, f, indent=2)
        
    print("\n✓ Done! Strict Physical Pipeline Structure Created at 'Strict_OOD_Pipeline/'")

if __name__ == "__main__":
    main()
