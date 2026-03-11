import json
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Load all 4 quadrant models' results
    
    # 1. BOTH (Ours: Static + Dynamic)
    with open("ood_results_summary.json", "r") as f:
        pg_res = json.load(f)
        
    # 2. NEITHER (Clean Only / No Curr)
    with open("ood_results_summary_no_curr.json", "r") as f:
        no_curr_res = json.load(f)
        
    # 3. DYNAMIC ONLY (Direct Artifacts)
    with open("ood_results_summary_direct_art.json", "r") as f:
        direct_art_res = json.load(f)
        
    # 4. STATIC ONLY
    try:
        with open("ood_results_summary_static_augs_only.json", "r") as f:
            static_only_res = json.load(f)
    except FileNotFoundError:
        print("Waiting for training and evaluation of Static-Only model to complete.")
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
    
    sa_clean = static_only_res["Clean"]
    sa_seen = static_only_res["Seen_Average"]
    sa_unseen = static_only_res["Unseen_Average"]
    
    severities = range(1, 6)
    
    # 1. Plot SEEN ARTIFACTS Comparison
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(severities, [pg_clean]*5, 'k--', label='Clean Upper Bound (No Noise)', alpha=0.5)
    
    plt.plot(severities, pg_seen, 'b-o', label='BOTH: Full PG-CNN (Static + Dynamic)', linewidth=3)
    plt.plot(severities, da_seen, color='green', linestyle='-.', marker='s', label='DYNAMIC ONLY (Direct Artifacts - Skip Stg 1 & 2)', linewidth=2.5)
    plt.plot(severities, sa_seen, color='orange', linestyle='-', marker='D', label='STATIC ONLY (Physics + Stoch Augs - No Dynamic OOD)', linewidth=2.5)
    plt.plot(severities, nc_seen, color='red', linestyle='--', marker='^', label='NEITHER (Clean Only - No Curriculum)', linewidth=2)
    
    plt.title("Robustness to SEEN Artifacts\n(Augmentation Syngergy Ablation)", fontsize=13)
    plt.xlabel("Severity Level", fontsize=11)
    plt.ylabel("Accuracy", fontsize=11)
    plt.ylim([0, 1.05])
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower left', fontsize=9)
    
    # 2. Plot UNSEEN OOD Comparison
    plt.subplot(1, 2, 2)
    plt.plot(severities, [pg_clean]*5, 'k--', label='Clean Upper Bound (No Noise)', alpha=0.5)
    
    plt.plot(severities, pg_unseen, 'b-o', label='BOTH: Full PG-CNN (Static + Dynamic)', linewidth=3)
    plt.plot(severities, da_unseen, color='green', linestyle='-.', marker='s', label='DYNAMIC ONLY (Direct Artifacts - Skip Stg 1 & 2)', linewidth=2.5)
    plt.plot(severities, sa_unseen, color='orange', linestyle='-', marker='D', label='STATIC ONLY (Physics + Stoch Augs - No Dynamic OOD)', linewidth=2.5)
    plt.plot(severities, nc_unseen, color='red', linestyle='--', marker='^', label='NEITHER (Clean Only - No Curriculum)', linewidth=2)
    
    plt.title("Robustness to UNSEEN OOD Corruptions\n(Augmentation Synergy Ablation)", fontsize=13)
    plt.xlabel("Severity Level", fontsize=11)
    plt.ylabel("Accuracy", fontsize=11)
    plt.ylim([0, 1.05])
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower left', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('ood_augmentation_synergy_comparison.png', dpi=300)
    print("✓ Saved Augmentation Synergy Ablation plot to ood_augmentation_synergy_comparison.png")
    
    # Print numerical comparison
    print("\n--- Augmentation Synergy Summary Comparison ---\n")
    print("                 | Static Augs | Dynamic OOD | Mean Seen | Mean Unseen")
    print("-----------------|-------------|-------------|-----------|------------")
    print(f"Full PG-CNN      |     YES     |     YES     | {np.mean(pg_seen)*100:6.2f}%  | {np.mean(pg_unseen)*100:6.2f}%")
    print(f"Dynamic Only     |     NO      |     YES     | {np.mean(da_seen)*100:6.2f}%  | {np.mean(da_unseen)*100:6.2f}%")
    print(f"Static Only      |     YES     |     NO      | {np.mean(sa_seen)*100:6.2f}%  | {np.mean(sa_unseen)*100:6.2f}%")
    print(f"Neither (Clean)  |     NO      |     NO      | {np.mean(nc_seen)*100:6.2f}%  | {np.mean(nc_unseen)*100:6.2f}%")

if __name__ == "__main__":
    main()
