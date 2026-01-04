"""
Visualization script for Information Autopsy diagnostics.
Plots per-feature information costs across beta sweep.
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_info_autopsy(results_dir: Path, output_path: Path = None):
    """
    Plot information autopsy results from beta sweep.
    
    Args:
        results_dir: Directory containing beta sweep results (JSON files)
        output_path: Where to save the plot (optional)
    """
    # Collect results
    beta_values = []
    info_dim_0 = []  # Delta
    info_dim_1 = []  # Micro
    info_total = []
    
    # Find all result files
    result_files = sorted(results_dir.glob("**/training_history.json"))
    
    if not result_files:
        print(f"No training_history.json files found in {results_dir}")
        return
    
    for result_file in result_files:
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
            
            # Extract beta from parent directory name or config
            # Assuming directory structure like: runs/beta_0.1/training_history.json
            beta_str = result_file.parent.name.replace("beta_", "")
            beta = float(beta_str)
            
            # Get final epoch values
            if data and len(data) > 0:
                final_entry = data[-1]
                
                beta_values.append(beta)
                info_dim_0.append(final_entry.get("info_dim_0", 0))
                info_dim_1.append(final_entry.get("info_dim_1", 0))
                info_total.append(final_entry.get("info_total", 0))
        except Exception as e:
            print(f"Error processing {result_file}: {e}")
            continue
    
    if not beta_values:
        print("No valid data found")
        return
    
    # Sort by beta
    sorted_indices = np.argsort(beta_values)
    beta_values = np.array(beta_values)[sorted_indices]
    info_dim_0 = np.array(info_dim_0)[sorted_indices]
    info_dim_1 = np.array(info_dim_1)[sorted_indices]
    info_total = np.array(info_total)[sorted_indices]
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Individual components
    ax1.plot(beta_values, info_dim_0, 'o-', label='info_dim_0 (Delta)', 
             color='#2E86AB', linewidth=2, markersize=8)
    ax1.plot(beta_values, info_dim_1, 's-', label='info_dim_1 (Micro)', 
             color='#A23B72', linewidth=2, markersize=8)
    ax1.plot(beta_values, info_total, '^--', label='Total Info Cost', 
             color='#F18F01', linewidth=1.5, markersize=6, alpha=0.7)
    
    ax1.set_xlabel('Beta (β)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Information Cost (KL Divergence)', fontsize=12, fontweight='bold')
    ax1.set_title('Information Autopsy: Feature-Level KL Tracking', fontsize=14, fontweight='bold')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=10, framealpha=0.9)
    
    # Add annotation for sweet spot
    if len(beta_values) > 2:
        # Find where micro drops significantly
        micro_ratio = info_dim_1 / (info_dim_0 + 1e-10)
        sweet_spot_idx = np.argmin(micro_ratio)
        sweet_beta = beta_values[sweet_spot_idx]
        
        ax1.axvline(sweet_beta, color='red', linestyle=':', linewidth=2, alpha=0.5)
        ax1.text(sweet_beta, ax1.get_ylim()[1] * 0.5, 
                f'Sweet Spot\nβ={sweet_beta:.3f}',
                ha='center', va='center', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Plot 2: Ratio analysis
    ratio = info_dim_1 / (info_dim_0 + 1e-10)
    ax2.plot(beta_values, ratio, 'o-', color='#6A4C93', linewidth=2, markersize=8)
    ax2.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='Equal Usage')
    
    ax2.set_xlabel('Beta (β)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Micro / Delta Ratio', fontsize=12, fontweight='bold')
    ax2.set_title('Feature Selection: Micro vs Delta Usage', fontsize=14, fontweight='bold')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    else:
        plt.savefig('info_autopsy_diagnostic.png', dpi=300, bbox_inches='tight')
        print("Saved plot to info_autopsy_diagnostic.png")
    
    plt.show()
    
    # Print summary
    print("\n" + "="*60)
    print("INFORMATION AUTOPSY SUMMARY")
    print("="*60)
    for i, beta in enumerate(beta_values):
        print(f"β = {beta:8.4f} | Delta: {info_dim_0[i]:8.4f} | Micro: {info_dim_1[i]:8.4f} | Ratio: {ratio[i]:8.4f}")
    print("="*60)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        results_dir = Path(sys.argv[1])
    else:
        results_dir = Path("runs")
    
    output_path = Path("diagnostics/info_autopsy_diagnostic.png")
    output_path.parent.mkdir(exist_ok=True)
    
    plot_info_autopsy(results_dir, output_path)
