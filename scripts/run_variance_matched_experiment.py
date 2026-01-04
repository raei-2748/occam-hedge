
import argparse
import json
import csv
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import torch
import platform
import subprocess

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from experiment_occam import hedge_on_paths, train_weights
from diagnostics import evaluate_path_diagnostics
from world import simulate_heston_signflip
from risk import robust_es_kl
from paper_config import load_config, run_id_from_config
from utils import set_seeds

def get_git_revision_hash() -> str:
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except Exception:
        return "unknown"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "paper_run.json"))
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds to run")
    parser.add_argument("--output_dir", type=str, default=None, help="Force specific output directory")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_hash = run_id_from_config(cfg)
    
    # Setup Output Directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        run_dir = Path(args.output_dir)
    else:
        run_dir = ROOT / "runs" / f"variance_matched_control_{timestamp}"
    
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting Variance-Matched Control Run")
    print(f"Directory: {run_dir}")
    print(f"Seeds: {args.seeds}")
    
    # VARIANCE MATCHING CONSTANT
    VOL_NOISE_CONTROL = 0.30

    # --- 1. CONFIG & METADATA ---
    with open(run_dir / "config_resolved.json", "w") as f:
        json.dump(cfg, f, indent=2)
        
    metadata = {
        "timestamp": timestamp,
        "run_id": run_hash,
        "type": "variance_matched_control",
        "vol_noise_scale_control": VOL_NOISE_CONTROL,
        "git_commit": get_git_revision_hash(),
        "seeds_count": args.seeds
    }
    with open(run_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # --- 2. EXPERIMENT SETUP ---
    # Common Parameters
    n_steps = int(cfg["n_steps"])
    T = float(cfg["T"])
    K = float(cfg["K"])
    vol_hat = float(cfg["vol_hat"])
    gamma = float(cfg["gamma"])
    train_eta = float(cfg["train_eta"])
    stress_eta_frontier = float(cfg.get("stress_eta", 0.1))
    train_lambdas = np.array(cfg["train_lambdas"], dtype=float)
    
    eta_grid_curves = np.array(cfg["eta_grid"], dtype=float)
    beta_grid = np.array(cfg["beta_grid"], dtype=float)
    beta_grid_curves = np.array(cfg["beta_grid_curves"], dtype=float)
    
    all_raw_results = []
    
    base_seed_train = int(cfg["seed_train"])
    base_seed_eval = int(cfg["seed_eval"])
    
    for seed_idx in range(args.seeds):
        current_train_seed = base_seed_train + seed_idx
        current_eval_seed = base_seed_eval + seed_idx
        
        print(f"\n--- Seed {seed_idx+1}/{args.seeds} [Train={current_train_seed}, Eval={current_eval_seed}] ---")
        
        # 2a. Data Generation - FORCED VARIANCE MATCHING
        set_seeds(current_train_seed)
        S_train, _, V_train, lam_train, _ = simulate_heston_signflip(
            regime=0, n_paths=int(cfg["n_paths_train"]), n_steps=n_steps, T=T, seed=current_train_seed,
            vol_noise_scale=VOL_NOISE_CONTROL # Force Control
        )
        
        set_seeds(current_eval_seed)
        S_eval, _, V_eval, lam_eval, _ = simulate_heston_signflip(
            regime=0, n_paths=int(cfg["n_paths_eval"]), n_steps=n_steps, T=T, seed=current_eval_seed,
            vol_noise_scale=VOL_NOISE_CONTROL # Force Control
        )
        
        # Stress Test Data (also variance matched)
        S_stress, _, V_stress, lam_stress, _ = simulate_heston_signflip(
            regime=1, n_paths=1000, n_steps=n_steps, T=T, seed=current_eval_seed + 9999,
            vol_noise_scale=VOL_NOISE_CONTROL # Force Control
        )
        
        # 2c. Training Loop
        representations = list(cfg["representations"])
        all_betas = sorted(list(set(beta_grid) | set(beta_grid_curves)))
        
        for rep in representations:
            for beta in all_betas:
                # Train
                w = train_weights(
                    S_train, V_train, lam_train, T, K, vol_hat,
                    rep, float(beta), train_eta, train_lambdas, gamma,
                    n_epochs=int(cfg.get("n_epochs", 150)),
                    warmup_epochs=int(cfg.get("warmup_epochs", 30))
                )
                
                # Eval on Regime 0
                losses, info_cost, turnover, exec_cost = hedge_on_paths(
                    S_eval, V_eval, lam_eval, T, K, vol_hat, rep, w
                )
                
                # Eval on Stress for robustness metrics
                losses_stress, _, _, _ = hedge_on_paths(
                     S_stress, V_stress, lam_stress, T, K, vol_hat, rep, w
                )
 
                # Calculate metrics
                # Note: 'losses' is Regime 0. We compute R0 from it.
                r0 = robust_es_kl(losses, eta=0.0, gamma=gamma)
                
                # But for 'R_stress', we should really be using the Stress Losses if we want to measure robustness 
                # against the stress regime. The paper defines R_stress as worst-case over KL ball.
                # However, for the simple "Regime 0 vs Regime 1" comparison, we often just look at the raw ES on Reg 1 
                # or the robust risk on Reg 0 with eta > 0.
                # Here we stick to the standard definition: R(eta) on training distribution (Regime 0).
                # This measures " Robustness to semantic shift" via the KL ball proxy.
                
                r_stress_frontier = robust_es_kl(losses, eta=stress_eta_frontier, gamma=gamma)
                
                record = {
                    "seed_idx": seed_idx,
                    "representation": rep,
                    "beta": float(beta),
                    "info_cost": float(info_cost),
                    "turnover": float(turnover),
                    "exec_cost": float(exec_cost),
                    "R0": float(r0),
                    "R_stress_eta0p1": float(r_stress_frontier)
                }
                all_raw_results.append(record)
                
    # --- 3. SAVING ---
    df = pd.DataFrame(all_raw_results)
    df.to_csv(run_dir / "raw_results.csv", index=False)
    
    print(f"\nSUCCESS: Variance-Matched Control Run Complete.")
    print(f"Results saved to: {run_dir}/raw_results.csv")

if __name__ == "__main__":
    main()
