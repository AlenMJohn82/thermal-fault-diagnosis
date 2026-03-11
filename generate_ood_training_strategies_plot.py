import json
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Load all 3 models' results
    with open("ood_results_summary.json", "r") as f:
        pg_res = json.load(f)
        
    try:
        with open("ood_results_summary_no_curr.json", "r") as f:
            no_curr_res = json.load(f)
        with open("ood_results_summary_direct_art.json", "r") as f:
            direct_art_res = json.load(f)
    except FileNotFoundError:
        print("Waiting for training and evaluation of ablations to complete.")
        return

    # Extract averages
    pg_clean = pg_res["Clean"]
    pg_seen = pg_res["Seen_Average"]
    pg_unseen = pg_res["Unseen_Average"]
    
    nc_clean = no_curr_res["Clean"]
    nc_seen = no_curr_res["Seen_Average"]
    nc_unseen = no_curr_res["Unseen_Average"]
    
    da_clean = direct_art_res["Clean"]
    da_seen = direct_art_res["Seen_Average"]
    da_unseen = direct_art_res["Unseen_Average"]
    
    severities = range(1, 6)
    
    # 1. Plot SEEN ARTIFACTS Comparison
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(severities, [pg_clean]*5, 'k--', label='Clean Upper Bound (No Noise)', alpha=0.5)
    plt.plot(severities, pg_seen, 'b-o', label='Curriculum 3: PG-CNN (Full 3-Stage)', linewidth=3)
    plt.plot(severities, da_seen, color='green', linestyle='-.', marker='s', label='Curriculum 2: Direct Artifacts (Skip Stg 1 & 2)', linewidth=2.5)
    plt.plot(severities, nc_seen, color='red', linestyle='--', marker='^', label='Curriculum 1: No Curriculum (Clean Only)', linewidth=2)
    
    plt.title("Robustness to SEEN Artifacts\n(Training Strategy Ablation)", fontsize=13)
    plt.xlabel("Severity Level", fontsize=11)
    plt.ylabel("Accuracy", fontsize=11)
    plt.ylim([0, 1.05])
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower left', fontsize=10)
    
    # 2. Plot UNSEEN OOD Comparison
    plt.subplot(1, 2, 2)
    plt.plot(severities, [pg_clean]*5, 'k--', label='Clean Upper Bound (No Noise)', alpha=0.5)
    plt.plot(severities, pg_unseen, 'b-o', label='Curriculum 3: PG-CNN (Full 3-Stage)', linewidth=3)
    plt.plot(severities, da_unseen, color='green', linestyle='-.', marker='s', label='Curriculum 2: Direct Artifacts (Skip Stg 1 & 2)', linewidth=2.5)
    plt.plot(severities, nc_unseen, color='red', linestyle='--', marker='^', label='Curriculum 1: No Curriculum (Clean Only)', linewidth=2)
    
    plt.title("Robustness to UNSEEN OOD Corruptions\n(Training Strategy Ablation)", fontsize=13)
    plt.xlabel("Severity Level", fontsize=11)
    plt.ylabel("Accuracy", fontsize=11)
    plt.ylim([0, 1.05])
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('ood_training_strategy_comparison.png', dpi=300)
    print("✓ Saved Training Strategy Ablation plot to ood_training_strategy_comparison.png")
    
    # Print numerical comparison
    print("\n--- Training Strategy Summary Comparison ---")
    print("Strategy             | Mean Seen | Mean Unseen")
    print(f"3-Stage Curriculum   | {np.mean(pg_seen)*100:6.2f}%  | {np.mean(pg_unseen)*100:6.2f}%")
    print(f"Direct Artifacts     | {np.mean(da_seen)*100:6.2f}%  | {np.mean(da_unseen)*100:6.2f}%")
    print(f"Clean Only (No Curr) | {np.mean(nc_seen)*100:6.2f}%  | {np.mean(nc_unseen)*100:6.2f}%")

if __name__ == "__main__":
    main()
