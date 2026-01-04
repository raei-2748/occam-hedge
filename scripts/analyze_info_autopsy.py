"""
Analyze Information Autopsy logs to verify selective channel collapse.

This script examines training histories to confirm:
1. Microstructure channel (info_dim_1) collapses at high beta_micro
2. Price/Delta channel (info_dim_0) remains active
"""

import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def analyze_info_autopsy(run_dir):
    """Analyze training histories for selective channel collapse."""
    
    run_dir = Path(run_dir)
    history_files = sorted(run_dir.glob("**/training_history.json"))
    
    if not history_files:
        print(f"No training history files found in {run_dir}")
        return
    
    results = []
    
    for hist_file in history_files:
        with open(hist_file) as f:
            history = json.load(f)
        
        if not history:
            continue
        
        # Get final epoch values
        final = history[-1]
        
        # Extract beta values from directory name or history
        dir_name = hist_file.parent.name
        if "hierarchical" in final and final["hierarchical"]:
            beta_price = final.get("beta_price", 0.0)
            beta_micro = final.get("beta_micro", 0.0)
        else:
            # Parse from directory name for backward compat
            if "bp" in dir_name and "bm" in dir_name:
                parts = dir_name.split("_")
                beta_price = float([p for p in parts if p.startswith("bp")][0][2:])
                beta_micro = float([p for p in parts if p.startswith("bm")][0][2:])
            else:
                continue
        
        info_dim_0 = final.get("info_dim_0", 0.0)
        info_dim_1 = final.get("info_dim_1", 0.0)
        
        results.append({
            "beta_price": beta_price,
            "beta_micro": beta_micro,
            "info_dim_0_delta": info_dim_0,
            "info_dim_1_micro": info_dim_1,
            "ratio": info_dim_1 / (info_dim_0 + 1e-8)
        })
    
    if not results:
        print("No hierarchical results found")
        return
    
    # Sort by beta_micro
    results = sorted(results, key=lambda x: x["beta_micro"])
    
    print("\n=== Information Autopsy: Channel-Specific KL Costs ===\n")
    print(f"{'beta_micro':<12} {'beta_price':<12} {'Delta (dim_0)':<15} {'Micro (dim_1)':<15} {'Ratio':<10}")
    print("-" * 70)
    
    for r in results:
        print(f"{r['beta_micro']:<12.4f} {r['beta_price']:<12.4f} "
              f"{r['info_dim_0_delta']:<15.4f} {r['info_dim_1_micro']:<15.4f} "
              f"{r['ratio']:<10.4f}")
    
    # Verify selective collapse
    print("\n=== Selective Channel Collapse Verification ===")
    
    high_beta_results = [r for r in results if r["beta_micro"] >= 1.0]
    
    if high_beta_results:
        for r in high_beta_results:
            delta_active = r["info_dim_0_delta"] > 0.01
            micro_collapsed = r["info_dim_1_micro"] < 0.01
            
            status = "✓" if (delta_active and micro_collapsed) else "✗"
            print(f"{status} beta_micro={r['beta_micro']:.2f}: "
                  f"Delta={'ACTIVE' if delta_active else 'COLLAPSED'}, "
                  f"Micro={'COLLAPSED' if micro_collapsed else 'ACTIVE'}")
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    beta_micros = [r["beta_micro"] for r in results]
    deltas = [r["info_dim_0_delta"] for r in results]
    micros = [r["info_dim_1_micro"] for r in results]
    
    ax1.plot(beta_micros, deltas, 'o-', label='Delta Channel (dim_0)', color='blue', linewidth=2)
    ax1.plot(beta_micros, micros, 's-', label='Micro Channel (dim_1)', color='red', linewidth=2)
    ax1.set_xlabel('beta_micro', fontsize=12)
    ax1.set_ylabel('KL Divergence (nats)', fontsize=12)
    ax1.set_title('Selective Channel Collapse', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    ax1.axhline(y=0.01, color='gray', linestyle='--', alpha=0.5, label='Collapse threshold')
    
    ratios = [r["ratio"] for r in results]
    ax2.plot(beta_micros, ratios, 'o-', color='purple', linewidth=2)
    ax2.set_xlabel('beta_micro', fontsize=12)
    ax2.set_ylabel('Micro/Delta Ratio', fontsize=12)
    ax2.set_title('Channel Selectivity', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    out_file = ROOT / "figures" / "hierarchical_info_autopsy.png"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=200, bbox_inches='tight')
    print(f"\n✓ Saved visualization: {out_file}")
    
    return results


if __name__ == "__main__":
    run_dir = ROOT / "runs"
    analyze_info_autopsy(run_dir)
