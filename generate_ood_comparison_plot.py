import json
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Load PG-CNN Results
    with open("ood_results_summary.json", "r") as f:
        pg_res = json.load(f)
        
    # Load Baseline Results
    try:
        with open("ood_results_summary_baseline.json", "r") as f:
            base_res = json.load(f)
    except FileNotFoundError:
        print("Baseline results not found yet. Run eval_ood_baseline.py first.")
        return

    # Extract averages
    pg_clean = pg_res["Clean"]
    pg_seen = pg_res["Seen_Average"]
    pg_unseen = pg_res["Unseen_Average"]
    
    base_clean = base_res["Clean"]
    base_seen = base_res["Seen_Average"]
    base_unseen = base_res["Unseen_Average"]
    
    severities = range(1, 6)
    
    # 1. Plot SEEN ARTIFACTS Comparison
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(severities, [pg_clean]*5, 'k--', label='Clean Upper Bound', alpha=0.5)
    plt.plot(severities, pg_seen, 'b-o', label='Physics-Guided CNN (Ours)', linewidth=2.5)
    plt.plot(severities, base_seen, 'g-s', label='Baseline ResNet18', linewidth=2.5)
    plt.title("Robustness to SEEN Artifacts\n(Used in Training)", fontsize=12)
    plt.xlabel("Severity Level", fontsize=11)
    plt.ylabel("Accuracy", fontsize=11)
    plt.ylim([0, 1.05])
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 2. Plot UNSEEN OOD Comparison
    plt.subplot(1, 2, 2)
    plt.plot(severities, [pg_clean]*5, 'k--', label='Clean Upper Bound', alpha=0.5)
    plt.plot(severities, pg_unseen, 'r-o', label='Physics-Guided CNN (Ours)', linewidth=2.5)
    plt.plot(severities, base_unseen, 'm-s', label='Baseline ResNet18', linewidth=2.5)
    plt.title("Robustness to UNSEEN OOD Corruptions\n(Stress Test)", fontsize=12)
    plt.xlabel("Severity Level", fontsize=11)
    plt.ylabel("Accuracy", fontsize=11)
    plt.ylim([0, 1.05])
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('ood_baseline_comparison.png', dpi=300)
    print("✓ Saved comparison plot to ood_baseline_comparison.png")
    
    # Print numerical comparison
    print("\n--- Summary Comparison ---")
    print(f"Mean Seen   | PG-CNN: {np.mean(pg_seen)*100:.2f}% | Baseline: {np.mean(base_seen)*100:.2f}%")
    print(f"Mean Unseen | PG-CNN: {np.mean(pg_unseen)*100:.2f}% | Baseline: {np.mean(base_unseen)*100:.2f}%")

if __name__ == "__main__":
    main()
