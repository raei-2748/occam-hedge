"""
Occam Frontier Sweep - Variance-Matched Control.
Evaluates the information-robustness trade-off for the 'combined' representation.
"""
import argparse
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import torch
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from experiment_occam import hedge_on_paths, train_weights
from world import simulate_heston_signflip
from risk import robust_es_kl
from utils import set_seeds

# --- Configuration ---
VOL_NOISE_CONTROL = 0.30  # Variance Matching
BETA_GRID = [0.0, 0.01, 0.1, 1.0, 10.0]
REPRESENTATION = "combined"
GAMMA = 0.95


def compute_wrong_way_score(V, actions, regime):
    """
    Wrong-Way Trading Score (W_r):
    Measures misalignment between volume signal and trading aggressiveness.
    
    In Regime 0: High Volume -> Trade More (correct)
    In Regime 1: High Volume -> Trade Less (correct)
    
    W_r = Correlation(Volume, |Trade|) with regime-appropriate sign check.
    A positive W_r in Regime 1 indicates wrong-way trading.
    """
    V_flat = V.flatten()
    actions_flat = np.abs(actions.flatten())
    
    corr = np.corrcoef(V_flat, actions_flat)[0, 1]
    
    # In Regime 0, positive correlation is correct (high vol -> trade more)
    # In Regime 1, negative correlation is correct (high vol -> trade less)
    if regime == 0:
        wrong_way = -corr  # More negative = more wrong
    else:
        wrong_way = corr   # More positive = more wrong
        
    return wrong_way


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds")
    parser.add_argument("--n_epochs", type=int, default=150, help="Training epochs")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        run_dir = Path(args.output_dir)
    else:
        run_dir = ROOT / "runs" / f"occam_frontier_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("OCCAM FRONTIER SWEEP - Variance-Matched Control")
    print("=" * 60)
    print(f"Output: {run_dir}")
    print(f"Seeds: {args.seeds}")
    print(f"Betas: {BETA_GRID}")
    print(f"Vol Noise Scale: {VOL_NOISE_CONTROL} (matched)")
    print()
    
    # Fixed experiment parameters
    n_steps = 30
    T = 30 / 252
    K = 100.0
    vol_hat = 0.2
    train_eta = 0.0
    train_lambdas = [0.01]
    n_paths_train = 5000
    n_paths_eval = 2000
    
    all_results = []
    
    for seed_idx in range(args.seeds):
        train_seed = seed_idx
        eval_seed = 777 + seed_idx
        
        print(f"\n--- Seed {seed_idx + 1}/{args.seeds} ---")
        
        # Generate Regime 0 data (Variance Matched)
        set_seeds(train_seed)
        S_train, _, V_train, lam_train, _ = simulate_heston_signflip(
            regime=0, n_paths=n_paths_train, n_steps=n_steps, T=T, seed=train_seed,
            vol_noise_scale=VOL_NOISE_CONTROL
        )
        
        set_seeds(eval_seed)
        S_eval0, _, V_eval0, lam_eval0, _ = simulate_heston_signflip(
            regime=0, n_paths=n_paths_eval, n_steps=n_steps, T=T, seed=eval_seed,
            vol_noise_scale=VOL_NOISE_CONTROL
        )
        
        # Generate Regime 1 data (Variance Matched)
        S_eval1, _, V_eval1, lam_eval1, _ = simulate_heston_signflip(
            regime=1, n_paths=n_paths_eval, n_steps=n_steps, T=T, seed=eval_seed + 9999,
            vol_noise_scale=VOL_NOISE_CONTROL
        )
        
        for beta in BETA_GRID:
            print(f"  Training beta={beta}...")
            
            # Train
            w = train_weights(
                S_train, V_train, lam_train, T, K, vol_hat,
                REPRESENTATION, float(beta), train_eta, train_lambdas, GAMMA,
                n_epochs=args.n_epochs,
                warmup_epochs=30
            )
            
            # Eval Regime 0
            losses0, info_cost, turnover0, exec_cost0 = hedge_on_paths(
                S_eval0, V_eval0, lam_eval0, T, K, vol_hat, REPRESENTATION, w
            )
            R0 = np.percentile(losses0, 95)  # ES_0.95
            
            # Eval Regime 1 (Stress)
            losses1, _, turnover1, exec_cost1 = hedge_on_paths(
                S_eval1, V_eval1, lam_eval1, T, K, vol_hat, REPRESENTATION, w
            )
            R1 = np.percentile(losses1, 95)  # ES_0.95 under stress
            
            # Wrong-Way Trading Score (on Regime 1)
            # Use per-path mean volume vs per-path loss
            V_mean_per_path = V_eval1.mean(axis=1)  # (n_paths,)
            W1 = np.corrcoef(V_mean_per_path, losses1)[0, 1]
            
            record = {
                "seed": seed_idx,
                "beta": float(beta),
                "info_cost": float(info_cost),
                "R0": float(R0),
                "R1": float(R1),
                "turnover_0": float(turnover0),
                "turnover_1": float(turnover1),
                "wrong_way_score": float(W1)
            }
            all_results.append(record)
            print(f"    R0={R0:.2f}, R1={R1:.2f}, Info={info_cost:.4f}, W={W1:.3f}")
    
    # Save results
    df = pd.DataFrame(all_results)
    df.to_csv(run_dir / "paper_frontier.csv", index=False)
    print(f"\nSaved: {run_dir / 'paper_frontier.csv'}")
    
    # Aggregate by beta
    agg = df.groupby("beta").agg({
        "R0": ["mean", "std"],
        "R1": ["mean", "std"],
        "info_cost": ["mean", "std"],
        "wrong_way_score": ["mean", "std"]
    }).reset_index()
    agg.columns = ['_'.join(col).strip('_') for col in agg.columns.values]
    
    print("\n--- Aggregated Results ---")
    print(agg.to_string(index=False))
    
    # --- Generate Occam Frontier Plot ---
    print("\nGenerating Occam Frontier plot...")
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Color by info_cost
    scatter = ax.scatter(
        df["R0"], df["R1"],
        c=np.log10(df["info_cost"] + 1e-6),
        cmap="viridis",
        s=100,
        alpha=0.7,
        edgecolors="k"
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(r"$\log_{10}$ Info Cost $\mathcal{C}(\phi)$", fontsize=12)
    
    # Add beta annotations
    for beta in BETA_GRID:
        subset = df[df["beta"] == beta]
        mean_r0 = subset["R0"].mean()
        mean_r1 = subset["R1"].mean()
        ax.annotate(
            f"β={beta}",
            (mean_r0, mean_r1),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=9,
            fontweight="bold"
        )
    
    # Diagonal reference
    lims = [min(df["R0"].min(), df["R1"].min()) * 0.9,
            max(df["R0"].max(), df["R1"].max()) * 1.1]
    ax.plot(lims, lims, 'k--', alpha=0.3, label="R0 = R1")
    
    ax.set_xlabel(r"$R_0$ (In-Sample $ES_{0.95}$)", fontsize=12)
    ax.set_ylabel(r"$R_1$ (Out-of-Sample $ES_{0.95}$)", fontsize=12)
    ax.set_title("Occam Frontier: Information-Robustness Trade-off\n(Variance-Matched Control)", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(run_dir / "fig_occam_frontier.png", dpi=150)
    print(f"Saved: {run_dir / 'fig_occam_frontier.png'}")
    
    # Copy to figures/ for paper
    figures_dir = ROOT / "figures"
    figures_dir.mkdir(exist_ok=True)
    import shutil
    shutil.copy2(run_dir / "fig_occam_frontier.png", figures_dir / "fig_occam_frontier.png")
    shutil.copy2(run_dir / "paper_frontier.csv", ROOT / "paper_frontier.csv")
    print(f"Copied to: figures/fig_occam_frontier.png, paper_frontier.csv")
    
    print("\n✅ SUCCESS: Occam Frontier Sweep Complete.")

if __name__ == "__main__":
    main()
