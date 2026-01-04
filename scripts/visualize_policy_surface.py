"""
Policy Surface Visualization - "Smoking Gun" for Simplicity Bias

This script provides visual proof that high-beta models learn to ignore
confusing volume signals, producing flat surfaces invariant to V_t.

Usage:
    python scripts/visualize_policy_surface.py --low_beta_dir runs/beta_0.0100_combined --high_beta_dir runs/beta_1.0000_combined
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from policies import FactorizedVariationalPolicy
from experiment_occam import occam_features_torch


def load_model_from_checkpoint(checkpoint_dir: Path, representation: str):
    """Load trained model from checkpoint directory."""
    # Determine input dimension
    if representation == "combined":
        input_dim = 2
    elif representation == "micro":
        input_dim = 3
    elif representation == "oracle":
        input_dim = 4
    else:
        input_dim = 1
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FactorizedVariationalPolicy(input_dim, latent_dim_per_feature=2).to(device)
    
    # Find weights file (could be in different formats)
    weights_file = checkpoint_dir / "model_weights.pt"
    if not weights_file.exists():
        # Try to find any .pt file
        pt_files = list(checkpoint_dir.glob("*.pt"))
        if pt_files:
            weights_file = pt_files[0]
        else:
            raise FileNotFoundError(f"No model weights found in {checkpoint_dir}")
    
    model.load_state_dict(torch.load(weights_file, map_location=device))
    model.eval()
    
    return model, device


def generate_policy_surface(model, device, S_range, V_range, K, vol_hat, representation):
    """
    Generate policy surface over (S_t, V_t) grid.
    
    Returns:
        S_grid, V_grid, action_grid
    """
    S_grid, V_grid = np.meshgrid(S_range, V_range)
    action_grid = np.zeros_like(S_grid)
    
    # Fixed tau for visualization (mid-point)
    tau = 0.5
    
    with torch.no_grad():
        for i in range(S_grid.shape[0]):
            for j in range(S_grid.shape[1]):
                S_t = torch.tensor([S_grid[i, j]], dtype=torch.float32, device=device)
                V_t = torch.tensor([V_grid[i, j]], dtype=torch.float32, device=device)
                tau_t = torch.tensor([tau], dtype=torch.float32, device=device)
                
                # Generate features
                feats = occam_features_torch(representation, S_t, tau_t, V_t, K, vol_hat)
                
                # Get action (use mean of stochastic policy for visualization)
                action, _, _ = model(feats)
                action_grid[i, j] = action.item()
    
    return S_grid, V_grid, action_grid


def plot_policy_surfaces(
    S_grid, V_grid, action_low, action_high,
    low_beta, high_beta, output_path
):
    """
    Create side-by-side heatmaps showing policy surfaces.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Determine common color scale for fair comparison
    vmin = min(action_low.min(), action_high.min())
    vmax = max(action_low.max(), action_high.max())
    
    # Low beta plot
    im1 = ax1.contourf(S_grid, V_grid, action_low, levels=20, cmap='RdYlBu_r', vmin=vmin, vmax=vmax)
    ax1.contour(S_grid, V_grid, action_low, levels=10, colors='black', alpha=0.3, linewidths=0.5)
    ax1.set_xlabel('Spot Price $S_t$', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Volume Proxy $V_t$', fontsize=12, fontweight='bold')
    ax1.set_title(f'Low β ≈ {low_beta:.4f}\n(Reactive: Complex Surface)', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.2)
    plt.colorbar(im1, ax=ax1, label='Hedge Position $a$')
    
    # High beta plot
    im2 = ax2.contourf(S_grid, V_grid, action_high, levels=20, cmap='RdYlBu_r', vmin=vmin, vmax=vmax)
    ax2.contour(S_grid, V_grid, action_high, levels=10, colors='black', alpha=0.3, linewidths=0.5)
    ax2.set_xlabel('Spot Price $S_t$', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Volume Proxy $V_t$', fontsize=12, fontweight='bold')
    ax2.set_title(f'High β ≈ {high_beta:.4f}\n(Simplified: Flat in $V_t$)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.2)
    plt.colorbar(im2, ax=ax2, label='Hedge Position $a$')
    
    plt.suptitle('Policy Surface Visualization: Simplicity Bias Evidence', 
                 fontsize=15, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved policy surface visualization to {output_path}")
    
    # Compute and print statistics
    print("\n" + "="*60)
    print("POLICY SURFACE ANALYSIS")
    print("="*60)
    
    # Variance along V dimension (should be low for high beta)
    var_V_low = np.var(action_low, axis=0).mean()  # Variance across V for each S
    var_V_high = np.var(action_high, axis=0).mean()
    
    # Variance along S dimension (should remain high for both)
    var_S_low = np.var(action_low, axis=1).mean()
    var_S_high = np.var(action_high, axis=1).mean()
    
    print(f"Low β = {low_beta:.4f}:")
    print(f"  Variance along V_t: {var_V_low:.6f}")
    print(f"  Variance along S_t: {var_S_low:.6f}")
    print(f"  V/S variance ratio: {var_V_low/var_S_low:.6f}")
    
    print(f"\nHigh β = {high_beta:.4f}:")
    print(f"  Variance along V_t: {var_V_high:.6f}")
    print(f"  Variance along S_t: {var_S_high:.6f}")
    print(f"  V/S variance ratio: {var_V_high/var_S_high:.6f}")
    
    print(f"\nSimplicity Metric (V-variance reduction): {(1 - var_V_high/var_V_low)*100:.1f}%")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Visualize policy surfaces for low and high beta models")
    parser.add_argument("--low_beta_dir", type=str, required=True, help="Directory with low-beta checkpoint")
    parser.add_argument("--high_beta_dir", type=str, required=True, help="Directory with high-beta checkpoint")
    parser.add_argument("--representation", type=str, default="combined", help="Feature representation")
    parser.add_argument("--K", type=float, default=100.0, help="Strike price")
    parser.add_argument("--vol_hat", type=float, default=0.2, help="Implied volatility")
    parser.add_argument("--S_min", type=float, default=80.0, help="Min spot price")
    parser.add_argument("--S_max", type=float, default=120.0, help="Max spot price")
    parser.add_argument("--V_min", type=float, default=0.1, help="Min volume proxy")
    parser.add_argument("--V_max", type=float, default=5.0, help="Max volume proxy")
    parser.add_argument("--grid_size", type=int, default=50, help="Grid resolution")
    parser.add_argument("--output", type=str, default=None, help="Output path")
    
    args = parser.parse_args()
    
    low_beta_dir = Path(args.low_beta_dir)
    high_beta_dir = Path(args.high_beta_dir)
    
    # Extract beta values from directory names or training history
    def extract_beta(checkpoint_dir):
        # Try to extract from directory name
        dir_name = checkpoint_dir.name
        if "beta_" in dir_name:
            beta_str = dir_name.split("beta_")[1].split("_")[0]
            return float(beta_str)
        
        # Try to extract from training history
        history_file = checkpoint_dir / "training_history.json"
        if history_file.exists():
            with open(history_file, 'r') as f:
                history = json.load(f)
                # Beta should be constant across epochs, get from first entry
                # Actually, it's not in history, it's in the config
                pass
        
        return 0.0  # Default
    
    low_beta = extract_beta(low_beta_dir)
    high_beta = extract_beta(high_beta_dir)
    
    print(f"Loading low-beta model (β ≈ {low_beta:.4f}) from {low_beta_dir}")
    print(f"Loading high-beta model (β ≈ {high_beta:.4f}) from {high_beta_dir}")
    
    # Load models
    model_low, device = load_model_from_checkpoint(low_beta_dir, args.representation)
    model_high, _ = load_model_from_checkpoint(high_beta_dir, args.representation)
    
    # Generate grids
    S_range = np.linspace(args.S_min, args.S_max, args.grid_size)
    V_range = np.linspace(args.V_min, args.V_max, args.grid_size)
    
    print(f"Generating policy surface for low-beta model...")
    S_grid, V_grid, action_low = generate_policy_surface(
        model_low, device, S_range, V_range, args.K, args.vol_hat, args.representation
    )
    
    print(f"Generating policy surface for high-beta model...")
    _, _, action_high = generate_policy_surface(
        model_high, device, S_range, V_range, args.K, args.vol_hat, args.representation
    )
    
    # Plot
    output_path = args.output
    if output_path is None:
        output_path = ROOT / "diagnostics" / "policy_surface_comparison.png"
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    plot_policy_surfaces(
        S_grid, V_grid, action_low, action_high,
        low_beta, high_beta, output_path
    )
    
    plt.show()


if __name__ == "__main__":
    main()
