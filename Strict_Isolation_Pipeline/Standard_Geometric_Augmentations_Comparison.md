# Physics-Guided Augmentations vs. Standard Geometric Augmentations

To demonstrate the unique value of our proposed Physics-Guided Augmentations, we conducted a direct 1-to-1 comparison against a curriculum learning pipeline utilizing only standard, "Normal" image augmentations. This experiment was conducted using the exact same strict, physically isolated 80/20 data split.

## Methodology: Standard Augmentation Curriculum

We replicated the 3-Stage Curriculum structure, replacing our physics-informed masks and stochastic intensity shifts with purely geometric perturbations:

*   **Stage 1 — Separate (individual augmentations)**: Apply one geometric augmentation at a time.
    > "In the first stage, individual augmentation techniques such as rotation, flipping, cropping, and scaling were applied separately to enable the model to learn fundamental geometric variations."
*   **Stage 2 — Combined (multiple augmentations together)**: Apply combinations (rotation + flipping, cropping + scaling, rotation + cropping + scaling).
    > "In the second stage, multiple augmentations were applied in combination to introduce more complex spatial variations and improve robustness to compounded transformations."
*   **Stage 3 — Original (clean images)**: 
    > No augmentations were applied during the final tuning stage, forcing the network to stabilize on perfect lab conditions.

*(Note: The generated images for this experiment have been permanently saved to the `Strict_OOD_Pipeline/TRAIN/Normal_Aug_Separate` and `Normal_Aug_Combined` folders for inspection).*

---

## 📊 Evaluation & Results

We evaluated both models on the isolated Test set across all Level 1, Level 2, and Level 3 Severity tests.

| Model Strategy | Stage 1 | Stage 2 | Stage 3 (Dynamic Phase) | Clean (L1) | Seen OOD (L2) | Unseen OOD (L3) |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| **Normal Augmentations** | Geometric (Separate)| Geometric (Combined)| None (Clean Only) | 100.00% | 18.37% | 15.13% |
| **Physics-Guided Curriculum (Ours)**| Physics Separated | Stochastic Combined | Dynamic Severe Noise | 100.00% | **98.64%** | **58.64%** |

### 🧠 Conclusion for Paper
Standard geometric augmentations (rotation, zooming, cropping) are completely insufficient for handling real-world severe industrial degradation. While a network trained on geometric augmentations can perfectly classify clean images (100%), it catastrophically collapses when faced with actual sensor and environmental noise (dropping to 18.3%). 

Our physics-guided structural augmentations, when coupled with dynamic noise injection, provide specialized robustness that traditional Computer Vision augmentations simply cannot match.
