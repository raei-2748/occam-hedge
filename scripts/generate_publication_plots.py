
import os
import sys
import torch
import pandas as pd
import json
import numpy as np
from pathlib import Path

# Setup paths
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

import plotting
from policies import FactorizedVariationalPolicy
from features import get_feature_dim

def refresh_existing_figures(run_dir: Path):
    """Refreshes all existing paper figures with upgraded aesthetics."""
    print("Refreshing existing figures with publication aesthetics...")
    
    # 1. Frontier Plot
    frontier_path = run_dir / "paper_frontier.csv"
    if frontier_path.exists():
        df = pd.read_csv(frontier_path)
        plotting.plot_frontier_beta_sweep(df, ROOT / "figures" / "fig_frontier_beta_sweep.png")
    
    # 2. Semantic Flip
    flip_path = run_dir / "paper_semantic_flip_summary.json"
    if flip_path.exists():
        with open(flip_path, 'r') as f:
            flip_data = json.load(f)
        plotting.plot_semantic_flip_correlations(flip_data, ROOT / "figures" / "fig_semantic_flip_correlations.png")
    
    # 3. Robust Curves
    curves_path = run_dir / "paper_robust_curves.json"
    if curves_path.exists():
        with open(curves_path, 'r') as f:
            curves_data = json.load(f)
        plotting.plot_robust_risk_vs_eta(curves_data["results"], ROOT / "figures" / "fig_robust_risk_vs_eta_band.png", use_bands=True)

def generate_thermomaps(run_dir: Path):
    """Generates policy surface thermomaps for beta=0 and beta=0.5."""
    print("Generating Policy Surface Thermomaps...")
    
    checkpoints_dir = run_dir / "checkpoints"
    
    # Model configs
    models_to_plot = [
        {"beta": 0.0, "name": "combined_beta_0.0000_seed_0"},
        {"beta": 0.5, "name": "combined_beta_0.5000_seed_0"}
    ]
    
    for m_cfg in models_to_plot:
        cp_path = checkpoints_dir / m_cfg["name"] / "model_weights.pt"
        if not cp_path.exists():
            print(f"Skipping {m_cfg['name']}, checkpoint not found.")
            continue
            
        # Reconstruct model
        # Note: 20-seed run uses micro_lags=1, include_prev_action=True
        representation = "combined"
        micro_lags = 1
        include_prev_action = True
        
        input_dim = get_feature_dim(representation, include_prev_action=include_prev_action, micro_lags=micro_lags)
        model = FactorizedVariationalPolicy(input_dim, latent_dim_per_feature=8)
        model.load_state_dict(torch.load(cp_path, map_location="cpu"))
        
        # Plot
        out_name = f"fig_thermomap_beta_{m_cfg['beta']:.1f}.png"
        plotting.plot_policy_surface(
            model, 
            representation, 
            m_cfg["beta"], 
            ROOT / "figures" / out_name,
            micro_lags=micro_lags,
            include_prev_action=include_prev_action
        )

if __name__ == "__main__":
    # Latest production run directory
    run_dir = ROOT / "final_20seed_results" / "runs" / "paper_20260111_040234_eb9082ec"
    
    refresh_existing_figures(run_dir)
    generate_thermomaps(run_dir)
    
    print("\nVisual Excellence Check: COMPLETE.")
