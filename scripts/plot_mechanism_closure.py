#!/usr/bin/env python3
"""
TASK F: Mechanism Closure Visualizer

Produces a summary figure showing the VIB mechanism in action:
1. As beta increases, regime info in Z decreases (Regime Predictability).
2. As beta increases, robustness to leak shifts increases (Performance Gap).
3. As beta increases, wrong-way dependence decreases.

Inputs:
- results/regime_probe_vs_beta.json
- results/sweep_results.csv

Output:
- figures/mechanism_closure.png
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

def load_probe_results(path):
    if not path.exists():
        print(f"Warning: Probe results not found at {path}")
        return pd.DataFrame()
    with open(path, "r") as f:
        data = json.load(f)
    return pd.DataFrame(data)

def load_sweep_results(path):
    if not path.exists():
        print(f"Warning: Sweep results not found at {path}")
        return pd.DataFrame()
    return pd.read_csv(path)

def plot_mechanism_closure(probe_df, sweep_df, output_path, l2_df=None):
    """
    Plot mechanism closure with optional L2 control overlay.
    
    Args:
        probe_df: VIB probe results
        sweep_df: VIB sweep results
        output_path: Output file path
        l2_df: Optional L2 control summary results
    """
    # Set style
    plt.style.use('seaborn-v0_8-paper')
    fig, axes = plt.subplots(3, 1, figsize=(8, 12), sharex=True)
    
    # Common Beta Grid
    # We assume 'beta_micro' is the primary beta if hierarchical, or 'beta' otherwise
    # For alignment, we'll try to group by 'beta'
    
    # --- Panel 1: Regime Predictability vs Beta ---
    ax = axes[0]
    if not probe_df.empty:
        # Aggregate by beta
        agg_probe = probe_df.groupby("beta").agg({
            "probe_z_auc": ["mean", "std"],
            "probe_mu_auc": ["mean", "std"]
        })
        betas = agg_probe.index
        
        # Plot Z Probe
        mu = agg_probe["probe_z_auc"]["mean"]
        std = agg_probe["probe_z_auc"]["std"].fillna(0)
        ax.plot(betas, mu, 'o-', color='purple', label=r'Probe $P(R|z)$', linewidth=2)
        ax.fill_between(betas, mu - std, mu + std, color='purple', alpha=0.2)
        
        # Plot Mu Probe (optional comparison)
        mu2 = agg_probe["probe_mu_auc"]["mean"]
        std2 = agg_probe["probe_mu_auc"]["std"].fillna(0)
        ax.plot(betas, mu2, 's--', color='magenta', label=r'Probe $P(R|\mu)$', alpha=0.7)
        
        ax.axhline(0.5, color='gray', linestyle=':', label="Random Chnc")
        
        # Overlay L2 control if available
        if l2_df is not None and not l2_df.empty:
            l2_data = l2_df[l2_df["model_type"] == "L2"]
            if not l2_data.empty:
                ax.plot(l2_data["l2_lambda"], l2_data["probe_auc_mean"],
                        's--', color='brown', label='L2 Control', linewidth=2, markersize=6)
                ax.fill_between(l2_data["l2_lambda"], 
                               l2_data["probe_auc_mean"] - l2_data["probe_auc_std"].fillna(0),
                               l2_data["probe_auc_mean"] + l2_data["probe_auc_std"].fillna(0),
                               color='brown', alpha=0.15)
        
        ax.set_ylabel("Regime Predictability (AUC)")
        ax.set_title("A. Information Bottleneck: Regime Info in Latent Space")
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No Probe Data", ha='center', va='center', transform=ax.transAxes)

    # --- Panel 2: Robustness to Leak Shifts ---
    ax = axes[1]
    if not sweep_df.empty:
        # Filter for relevant columns
        # We want to compare R1 (matched) vs R1_leak_shifted / R1_leak_broken
        # Group by beta
        
        # Identify the beta column
        beta_col = "beta_micro" if "beta_micro" in sweep_df.columns and sweep_df["beta_micro"].nunique() > 1 else "beta"
        
        # Dynamic aggregation dict based on available columns
        agg_dict = {"R1": ["mean", "std"]}
        if "R1_leak_broken" in sweep_df.columns:
            agg_dict["R1_leak_broken"] = ["mean", "std"]
        if "R1_leak_shifted" in sweep_df.columns:
            agg_dict["R1_leak_shifted"] = ["mean", "std"]
            
        agg_sweep = sweep_df.groupby(beta_col).agg(agg_dict)
        betas = agg_sweep.index
        
        # Plot Baseline (Matched)
        mu_base = agg_sweep["R1"]["mean"]
        std_base = agg_sweep["R1"]["std"].fillna(0)
        ax.plot(betas, mu_base, 'o-', color='blue', label="Matched (Train=Test)", linewidth=2)
        ax.fill_between(betas, mu_base - std_base, mu_base + std_base, color='blue', alpha=0.1)
        
        # Plot Broken Leak
        if "R1_leak_broken" in agg_sweep.columns:
            mu_broken = agg_sweep["R1_leak_broken"]["mean"]
            std_broken = agg_sweep["R1_leak_broken"]["std"].fillna(0)
            ax.plot(betas, mu_broken, '^-', color='red', label=r"Broken Leak (Test $\phi=0$)", linewidth=2)
            ax.fill_between(betas, mu_broken - std_broken, mu_broken + std_broken, color='red', alpha=0.1)
        
        # Plot Shifted Leak
        if "R1_leak_shifted" in agg_sweep.columns:
            mu_shifted = agg_sweep["R1_leak_shifted"]["mean"]
            std_shifted = agg_sweep["R1_leak_shifted"]["std"].fillna(0)
            ax.plot(betas, mu_shifted, 'v--', color='orange', label=r"Shifted Leak (Test $\phi \uparrow$)", alpha=0.7)
        
        # Overlay L2 control if available
        if l2_df is not None and not l2_df.empty:
            l2_data = l2_df[l2_df["model_type"] == "L2"]
            if not l2_data.empty:
                ax.plot(l2_data["l2_lambda"], l2_data["R1_matched_mean"],
                        's--', color='brown', label='L2 (Matched)', linewidth=1.5, markersize=5)
                ax.plot(l2_data["l2_lambda"], l2_data["R1_broken_mean"],
                        '^--', color='darkred', label='L2 (Broken)', linewidth=1.5, markersize=5)
        
        ax.set_ylabel("Risk $R_1$ (Lower is Better)")
        ax.set_title("B. Robustness: Performance under Leak Shifts")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No Sweep Data", ha='center', va='center', transform=ax.transAxes)

    # --- Panel 3: Wrong-Way Score ---
    ax = axes[2]
    if not sweep_df.empty:
        agg_ww = sweep_df.groupby(beta_col).agg({
            "wrong_way_score": ["mean", "std"]
        })
        betas = agg_ww.index
        
        mu_ww = agg_ww["wrong_way_score"]["mean"]
        std_ww = agg_ww["wrong_way_score"]["std"].fillna(0)
        
        ax.plot(betas, mu_ww, 'd-', color='green', label="VIB", linewidth=2)
        ax.fill_between(betas, mu_ww - std_ww, mu_ww + std_ww, color='green', alpha=0.2)
        
        # Overlay L2 control if available
        if l2_df is not None and not l2_df.empty:
            l2_data = l2_df[l2_df["model_type"] == "L2"]
            if not l2_data.empty:
                ax.plot(l2_data["l2_lambda"], l2_data["wrong_way_corr_mean"],
                        's--', color='brown', label='L2 Control', linewidth=2, markersize=6)
                ax.fill_between(l2_data["l2_lambda"],
                               l2_data["wrong_way_corr_mean"] - l2_data["wrong_way_corr_std"].fillna(0),
                               l2_data["wrong_way_corr_mean"] + l2_data["wrong_way_corr_std"].fillna(0),
                               color='brown', alpha=0.15)
        
        ax.axhline(0.0, color='gray', linestyle='--')
        ax.set_ylabel("Wrong-Way Correlation")
        ax.set_title("C. Mechanism: Disconnecting from Toxic Micro Signals")
        ax.set_xlabel(r"Information Penalty $\beta$ (log scale)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log") # Log scale for beta usually makes sense
    else:
        ax.text(0.5, 0.5, "No Data", ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"Saved figure to {output_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Plot mechanism closure with optional L2 overlay")
    parser.add_argument("--include_l2", action="store_true",
                        help="Overlay L2 control results from regularization_control_summary.csv")
    args = parser.parse_args()
    
    results_dir = ROOT / "results"
    probe_path = results_dir / "regime_probe_vs_beta.json"
    sweep_path = results_dir / "sweep_results.csv"  # or whatever main_sweep saves to
    
    # Check if files exist, if not try default specific filename or warn
    # main_sweep saves to runs/run_id/sweep_results.csv and copies to root/paper_sweep_results.csv
    # Let's try root/paper_sweep_results.csv as fallback
    if not sweep_path.exists():
        sweep_path = ROOT / "paper_sweep_results.csv"
        
    print(f"Loading probe data from: {probe_path}")
    print(f"Loading sweep data from: {sweep_path}")
    
    probe_df = load_probe_results(probe_path)
    sweep_df = load_sweep_results(sweep_path)
    
    # Load L2 control if requested
    l2_df = None
    if args.include_l2:
        l2_path = results_dir / "regularization_control_summary.csv"
        if l2_path.exists():
            print(f"Loading L2 control data from: {l2_path}")
            l2_df = pd.read_csv(l2_path)
        else:
            print(f"Warning: L2 control data not found at {l2_path}")
    
    output_path = ROOT / "figures" / "mechanism_closure.png"
    plot_mechanism_closure(probe_df, sweep_df, output_path, l2_df=l2_df)

if __name__ == "__main__":
    main()
