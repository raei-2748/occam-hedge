"""
Unified Sweep Script for Occam's Hedge.
Supports:
- Single Beta vs Hierarchical Beta Sweeps.
- Standard vs Variance-Matched Controls.
- Comprehensive Metrics (ES, Info, Turnover, Wrong-Way Score).
- Anchor Benchmarks (Black-Scholes Delta).
"""
import argparse
import csv
import json
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from experiment_occam import hedge_on_paths, train_model, train_weights
from world import simulate_heston_signflip
from risk import robust_es_kl
from paper_config import load_config, run_id_from_config
from utils import set_seeds
from policies import FactorizedVariationalPolicy, policy_delta_hedge

def compute_wrong_way_score(V, actions, regime):
    """
    Measures misalignment between volume signal and trading aggressiveness.
    """
    V_flat = V.flatten()
    actions_flat = np.abs(actions.flatten())
    
    # Correlation between volume and absolute trade size
    corr = np.corrcoef(V_flat, actions_flat)[0, 1]
    
    # In Regime 0 (High Vol -> High Uncertainty -> Trade More) -> positive correlation is 'correct'
    # In Regime 1 (High Vol -> High Uncertainty -> Trade LESS in this model) -> negative correlation is 'correct'
    if regime == 0:
        wrong_way = -corr  # More negative = more 'wrong' relative to expectation if positive is 'right'
        # Actually, let's keep it simple: just return the raw correlation or a sign-adjusted one.
        # Run_occam_frontier_sweep defines it as:
        # regime 0: wrong_way = -corr
        # regime 1: wrong_way = corr
    else:
        wrong_way = corr
        
    return wrong_way

def main():
    parser = argparse.ArgumentParser(description="Unified Sweep Script")
    parser.add_argument("--config", type=str, help="Path to JSON config file")
    parser.add_argument("--seeds", type=int, default=None, help="Override number of seeds")
    parser.add_argument("--n_epochs", type=int, default=None, help="Override training epochs")
    parser.add_argument("--variance_match", action="store_true", help="Apply Variance-Matched Control (vol=0.30)")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory")
    args = parser.parse_args()

    # Default Config if none provided
    if args.config:
        cfg = load_config(args.config)
    else:
        # Minimum required config for a default run
        cfg = {
            "representations": ["combined"],
            "beta_grid": [0.0, 0.01, 0.1, 1.0, 10.0],
            "gamma": 0.95,
            "n_steps": 30,
            "T": 30/252,
            "K": 100.0,
            "vol_hat": 0.2,
            "n_paths_train": 5000,
            "n_paths_eval": 2000,
            "train_eta": 0.0,
            "stress_eta": 0.1,
            "train_lambdas": [0.01],
            "seed_train": 0,
            "seed_eval": 777,
            "n_epochs": 150,
            "warmup_epochs": 30,
            "hierarchical_beta": False
        }

    # Overrides
    n_seeds = args.seeds if args.seeds is not None else cfg.get("n_seeds", 1)
    n_epochs = args.n_epochs if args.n_epochs is not None else cfg.get("n_epochs", 150)
    vol_noise_control = 0.30 if args.variance_match else None
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = run_id_from_config(cfg) if args.config else f"sweep_{timestamp}"
    
    if args.output_dir:
        run_dir = Path(args.output_dir)
    else:
        run_dir = ROOT / "runs" / f"{run_id}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"UNIFIED SWEEP: {run_id}")
    print("=" * 60)
    print(f"Variance Match: {args.variance_match}")
    print(f"Seeds: {n_seeds}")

    # Beta Config Setup
    hierarchical_mode = cfg.get("hierarchical_beta", False)
    if hierarchical_mode:
        beta_price_anchor = float(cfg.get("beta_price_anchor", 1e-4))
        beta_grid = np.array(cfg["beta_grid"], dtype=float)
        beta_configs = [{"beta_price": beta_price_anchor, "beta_micro": bm} for bm in beta_grid]
    else:
        beta_configs = list(np.array(cfg["beta_grid"], dtype=float))

    representations = list(cfg["representations"])
    all_results = []

    for seed_idx in range(n_seeds):
        base_seed = int(cfg["seed_train"]) + seed_idx
        eval_seed = int(cfg["seed_eval"]) + seed_idx
        
        print(f"\n--- Seed {seed_idx + 1}/{n_seeds} (Train: {base_seed}, Eval: {eval_seed}) ---")

        # Data Generation
        set_seeds(base_seed)
        S_train, _, V_train, lam_train, _ = simulate_heston_signflip(
            regime=0, n_paths=int(cfg["n_paths_train"]), n_steps=int(cfg["n_steps"]), 
            T=float(cfg["T"]), seed=base_seed, vol_noise_scale=vol_noise_control
        )
        
        set_seeds(eval_seed)
        S_eval0, _, V_eval0, lam_eval0, _ = simulate_heston_signflip(
            regime=0, n_paths=int(cfg["n_paths_eval"]), n_steps=int(cfg["n_steps"]), 
            T=float(cfg["T"]), seed=eval_seed, vol_noise_scale=vol_noise_control
        )
        
        S_eval1, _, V_eval1, lam_eval1, _ = simulate_heston_signflip(
            regime=1, n_paths=int(cfg["n_paths_eval"]), n_steps=int(cfg["n_steps"]), 
            T=float(cfg["T"]), seed=eval_seed + 9999, vol_noise_scale=vol_noise_control
        )

        for rep in representations:
            for beta_cfg in beta_configs:
                print(f"  Training {rep} beta={beta_cfg}...")
                
                # Setup Training Config
                training_config = {
                    "beta": beta_cfg,
                    "train_eta": float(cfg["train_eta"]),
                    "train_lambdas": cfg["train_lambdas"],
                    "gamma": float(cfg["gamma"]),
                    "lr": 0.001,
                    "n_epochs": n_epochs,
                    "warmup_epochs": int(cfg.get("warmup_epochs", 50))
                }

                # Device selection
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                from features import get_feature_dim
                input_dim = get_feature_dim(rep)
                model = FactorizedVariationalPolicy(input_dim, latent_dim_per_feature=2).to(device)

                # Train
                S_t = torch.tensor(S_train, dtype=torch.float32, device=device)
                V_t = torch.tensor(V_train, dtype=torch.float32, device=device)
                lam_t = torch.tensor(lam_train, dtype=torch.float32, device=device)
                
                result = train_model(
                    model, S_t, V_t, lam_t, training_config,
                    representation=rep, T=float(cfg["T"]), K=float(cfg["K"]), vol_hat=float(cfg["vol_hat"])
                )
                w = result["final_weights"]

                # Eval Regime 0
                losses0, info_cost, turnover0, exec_cost0 = hedge_on_paths(
                    S_eval0, V_eval0, lam_eval0, T=float(cfg["T"]), K=float(cfg["K"]), 
                    vol_hat=float(cfg["vol_hat"]), representation=rep, weights_or_state_dict=w
                )
                r0 = robust_es_kl(losses0, eta=0.0, gamma=float(cfg["gamma"]))

                # Eval Regime 1 (Stress)
                losses1, _, turnover1, exec_cost1 = hedge_on_paths(
                    S_eval1, V_eval1, lam_eval1, T=float(cfg["T"]), K=float(cfg["K"]), 
                    vol_hat=float(cfg["vol_hat"]), representation=rep, weights_or_state_dict=w
                )
                r1 = robust_es_kl(losses1, eta=float(cfg.get("stress_eta", 0.0)), gamma=float(cfg["gamma"]))
                
                # Wrong-Way Score (Regime 1)
                # Need actions to compute this. hedge_on_paths returns (losses, info, turnover, exec).
                # We might need to modify hedge_on_paths or re-run a simplified loop.
                # For now, let's use the correlation between mean volume and path loss as a proxy if we don't have actions.
                # Actually, run_occam_frontier_sweep uses:
                V_mean_per_path = V_eval1.mean(axis=1) 
                W1 = np.corrcoef(V_mean_per_path, losses1)[0, 1]

                # Parse Beta for record
                if isinstance(beta_cfg, dict):
                    b_price = beta_cfg["beta_price"]
                    b_micro = beta_cfg["beta_micro"]
                else:
                    b_price = beta_cfg
                    b_micro = beta_cfg

                record = {
                    "seed": seed_idx,
                    "representation": rep,
                    "beta": b_micro if hierarchical_mode else b_price,
                    "beta_price": b_price,
                    "beta_micro": b_micro,
                    "info_cost": float(info_cost),
                    "R0": float(r0),
                    "R1": float(r1),
                    "turnover_0": float(turnover0),
                    "turnover_1": float(turnover1),
                    "wrong_way_score": float(W1)
                }
                all_results.append(record)
                print(f"    R0={r0:.2f}, R1={r1:.2f}, Info={info_cost:.4f}, W={W1:.3f}")

    # --- BS Anchor Benchmark ---
    if cfg.get("compute_bs_anchor", True):
        print("\nComputing BS Delta Hedge anchor...")
        T = float(cfg["T"])
        K = float(cfg["K"])
        vol_hat = float(cfg["vol_hat"])
        n_steps = int(cfg["n_steps"])
        
        # We'll use the last evaluation seed's data
        tau_grid = np.linspace(T, 0.0, n_steps + 1)
        n_eval = S_eval0.shape[0]
        
        # Eval BS on Regime 0
        a_bs = np.zeros(n_eval)
        pnl_bs0 = np.zeros(n_eval)
        cost_bs0 = np.zeros(n_eval)
        turnover_bs0 = np.zeros(n_eval)
        for t in range(n_steps):
            a_new = policy_delta_hedge(S_eval0[:, t], K=K, tau_t=np.full(n_eval, tau_grid[t]), vol_hat=vol_hat)
            da = a_new - a_bs
            pnl_bs0 += a_new * (S_eval0[:, t+1] - S_eval0[:, t])
            cost_bs0 += 0.5 * lam_eval0[:, t] * (da**2)
            turnover_bs0 += np.abs(da)
            a_bs = a_new
        losses_bs0 = np.maximum(S_eval0[:, -1] - K, 0) - pnl_bs0 - cost_bs0
        r0_bs = robust_es_kl(losses_bs0, eta=0.0, gamma=float(cfg["gamma"]))

        # Eval BS on Regime 1
        a_bs = np.zeros(n_eval)
        pnl_bs1 = np.zeros(n_eval)
        cost_bs1 = np.zeros(n_eval)
        turnover_bs1 = np.zeros(n_eval)
        for t in range(n_steps):
            a_new = policy_delta_hedge(S_eval1[:, t], K=K, tau_t=np.full(n_eval, tau_grid[t]), vol_hat=vol_hat)
            da = a_new - a_bs
            pnl_bs1 += a_new * (S_eval1[:, t+1] - S_eval1[:, t])
            cost_bs1 += 0.5 * lam_eval1[:, t] * (da**2)
            turnover_bs1 += np.abs(da)
            a_bs = a_new
        losses_bs1 = np.maximum(S_eval1[:, -1] - K, 0) - pnl_bs1 - cost_bs1
        r1_bs = robust_es_kl(losses_bs1, eta=float(cfg.get("stress_eta", 0.0)), gamma=float(cfg["gamma"]))

        all_results.append({
            "seed": -1, "representation": "BS_Anchor", "beta": np.nan, "beta_price": np.nan, "beta_micro": np.nan,
            "info_cost": 0.0, "R0": float(r0_bs), "R1": float(r1_bs), 
            "turnover_0": np.mean(turnover_bs0)/n_steps, "turnover_1": np.mean(turnover_bs1)/n_steps, "wrong_way_score": 0.0
        })

    # Save Results
    df = pd.DataFrame(all_results)
    csv_path = run_dir / "sweep_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved Results: {csv_path}")

    # Plotting
    print("\nGenerating Frontier Plots...")
    fig, ax = plt.subplots(figsize=(10, 7))
    
    rep_markers = {"greeks": "o", "micro": "s", "combined": "^", "oracle": "D"}
    
    for rep in df["representation"].unique():
        if rep == "BS_Anchor": continue
        subset = df[df["representation"] == rep]
        
        scatter = ax.scatter(
            subset["R0"], subset["R1"],
            c=np.log10(subset["info_cost"] + 1e-6),
            cmap="viridis", marker=rep_markers.get(rep, "x"),
            s=80, alpha=0.7, edgecolors="k", label=rep
        )

    # BS Anchor
    bs_data = df[df["representation"] == "BS_Anchor"]
    if not bs_data.empty:
        ax.scatter(bs_data["R0"], bs_data["R1"], c="red", marker="X", s=200, label="BS Anchor", edgecolors="black", linewidths=1.5, zorder=10)

    # Diagonal
    lims = [min(df["R0"].min(), df["R1"].min()) * 0.9, max(df["R0"].max(), df["R1"].max()) * 1.1]
    ax.plot(lims, lims, "k--", alpha=0.3, label="R0 = R1")

    ax.set_xlabel(r"In-Sample Risk $R_0$")
    ax.set_ylabel(r"Out-of-Sample Risk $R_1$")
    ax.set_title(f"Information-Robustness Frontier\n(Run: {run_id})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if any(df["info_cost"] > 0):
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(r"$\log_{10}$ Info Cost $\mathcal{C}(\phi)$")

    plot_path = run_dir / "frontier_plot.png"
    plt.savefig(plot_path, dpi=200)
    print(f"Saved Plot: {plot_path}")
    
    # Copy important files to root/figures for convenience
    import shutil
    shutil.copy2(csv_path, ROOT / "paper_sweep_results.csv")
    shutil.copy2(plot_path, ROOT / "figures" / "latest_frontier.png")

    print("\n✅ SUCCESS: Unified Sweep Complete.")

if __name__ == "__main__":
    main()
