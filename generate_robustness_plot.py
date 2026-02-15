import json
import matplotlib.pyplot as plt
import numpy as np

def generate_plot():
    # Load results
    try:
        with open('noise_robustness_results.json', 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error: noise_robustness_results.json not found. Run noise_test.py first.")
        return

    noise_levels = data['noise_levels']
    # Convert to percent
    pg_acc = [x * 100 for x in data['physics_guided']]
    base_acc = [x * 100 for x in data['baseline']]

    # Plot
    plt.figure(figsize=(10, 6))
    
    # Plot lines with markers
    plt.plot(noise_levels, pg_acc, 'o-', linewidth=3, label='Physics-Guided CNN (Ours)', color='#1f77b4')
    plt.plot(noise_levels, base_acc, 's--', linewidth=3, label='Baseline ResNet18', color='#d62728')

    # Highlight the gap
    plt.fill_between(noise_levels, pg_acc, base_acc, color='gray', alpha=0.1, label='Performance Gap')
    
    # Random guess line (1/11 classes = 9.09%)
    plt.axhline(y=9.09, color='gray', linestyle=':', linewidth=2, label='Random Guess (9.1%)')

    
    # Annotate specific points of interest
    idx_05 = noise_levels.index(0.05)
    plt.annotate(f'+{pg_acc[idx_05]-base_acc[idx_05]:.1f}%', 
                 (noise_levels[idx_05], (pg_acc[idx_05]+base_acc[idx_05])/2),
                 ha='center', va='center', fontweight='bold', color='black',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))

    idx_10 = noise_levels.index(0.10)
    plt.annotate(f'+{pg_acc[idx_10]-base_acc[idx_10]:.1f}%', 
                 (noise_levels[idx_10], (pg_acc[idx_10]+base_acc[idx_10])/2),
                 ha='center', va='center', fontweight='bold', color='black',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))

    # Styling
    plt.title('Performance Robustness under Image Noise', fontsize=16, fontweight='bold')
    plt.xlabel('Noise Level (Gaussian $\sigma$)', fontsize=14)
    plt.ylabel('Classification Accuracy (%)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.ylim(0, 105)
    plt.xlim(0, 0.5)
    
    # Save
    plt.tight_layout()
    plt.savefig('noise_robustness_plot.png', dpi=300)
    print("Plot saved as noise_robustness_plot.png")

if __name__ == "__main__":
    generate_plot()
