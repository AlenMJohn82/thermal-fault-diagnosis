import json
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Load all 5 models' results
    with open("ood_results_summary.json", "r") as f:
        pg_res = json.load(f)
    with open("ood_results_summary_baseline.json", "r") as f:
        base_res = json.load(f)
    with open("ood_results_summary_baseline_mask.json", "r") as f:
        mask_res = json.load(f)
    with open("ood_results_summary_baseline_phys.json", "r") as f:
        phys_res = json.load(f)
        
    try:
        with open("ood_results_summary_baseline_static.json", "r") as f:
            static_res = json.load(f)
    except FileNotFoundError:
        print("Waiting for Static Fusion results to complete. Run eval_ood_baseline_static.py first.")
        return

    # Extract averages
    pg_clean = pg_res["Clean"]
    pg_seen = pg_res["Seen_Average"]
    pg_unseen = pg_res["Unseen_Average"]
    
    base_clean = base_res["Clean"]
    base_seen = base_res["Seen_Average"]
    base_unseen = base_res["Unseen_Average"]
    
    mask_seen = mask_res["Seen_Average"]
    mask_unseen = mask_res["Unseen_Average"]
    
    phys_seen = phys_res["Seen_Average"]
    phys_unseen = phys_res["Unseen_Average"]
    
    static_seen = static_res["Seen_Average"]
    static_unseen = static_res["Unseen_Average"]
    
    severities = range(1, 6)
    
    # 1. Plot SEEN ARTIFACTS Comparison
    plt.figure(figsize=(16, 7))
    
    plt.subplot(1, 2, 1)
    plt.plot(severities, [pg_clean]*5, 'k--', label='Clean Upper Bound', alpha=0.5)
    plt.plot(severities, pg_seen, 'b-o', label='Physics-Guided CNN (Adaptive Alpha)', linewidth=3)
    plt.plot(severities, static_seen, color='red', linestyle='-.', marker='x', label='Ablation 3: Static Fusion (Mask+Phys, No Alpha)', linewidth=2.5)
    plt.plot(severities, mask_seen, color='orange', linestyle='--', marker='^', label='Ablation 1: Mask Only', linewidth=1.5, alpha=0.7)
    plt.plot(severities, phys_seen, color='purple', linestyle='--', marker='D', label='Ablation 2: Phys Only', linewidth=1.5, alpha=0.7)
    plt.plot(severities, base_seen, 'g-s', label='Baseline ResNet18', linewidth=2, alpha=0.8)
    
    plt.title("Robustness to SEEN Artifacts\n(Ablation Study)", fontsize=13)
    plt.xlabel("Severity Level", fontsize=11)
    plt.ylabel("Accuracy", fontsize=11)
    plt.ylim([0, 1.05])
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower left', fontsize=9)
    
    # 2. Plot UNSEEN OOD Comparison
    plt.subplot(1, 2, 2)
    plt.plot(severities, [pg_clean]*5, 'k--', label='Clean Upper Bound', alpha=0.5)
    plt.plot(severities, pg_unseen, 'b-o', label='Physics-Guided CNN (Adaptive Alpha)', linewidth=3)
    plt.plot(severities, static_unseen, color='red', linestyle='-.', marker='x', label='Ablation 3: Static Fusion (Mask+Phys, No Alpha)', linewidth=2.5)
    plt.plot(severities, mask_unseen, color='orange', linestyle='--', marker='^', label='Ablation 1: Mask Only', linewidth=1.5, alpha=0.7)
    plt.plot(severities, phys_unseen, color='purple', linestyle='--', marker='D', label='Ablation 2: Phys Only', linewidth=1.5, alpha=0.7)
    plt.plot(severities, base_unseen, 'g-s', label='Baseline ResNet18', linewidth=2, alpha=0.8)
    
    plt.title("Robustness to UNSEEN OOD Corruptions\n(Ablation Study)", fontsize=13)
    plt.xlabel("Severity Level", fontsize=11)
    plt.ylabel("Accuracy", fontsize=11)
    plt.ylim([0, 1.05])
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower left', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('ood_ablation_static_comparison.png', dpi=300)
    print("✓ Saved 5-way ultimate ablation plot to ood_ablation_static_comparison.png")
    
    # Print numerical comparison
    print("\n--- Ultimate Ablation Summary Comparison ---")
    print("Model            | Mean Seen | Mean Unseen")
    print(f"PG-CNN (Ours)    | {np.mean(pg_seen)*100:6.2f}%  | {np.mean(pg_unseen)*100:6.2f}%")
    print(f"Ablation 3 Static| {np.mean(static_seen)*100:6.2f}%  | {np.mean(static_unseen)*100:6.2f}%")
    print(f"Ablation 1 Mask  | {np.mean(mask_seen)*100:6.2f}%  | {np.mean(mask_unseen)*100:6.2f}%")
    print(f"Ablation 2 Phys  | {np.mean(phys_seen)*100:6.2f}%  | {np.mean(phys_unseen)*100:6.2f}%")
    print(f"Baseline ResNet  | {np.mean(base_seen)*100:6.2f}%  | {np.mean(base_unseen)*100:6.2f}%")

if __name__ == "__main__":
    main()
