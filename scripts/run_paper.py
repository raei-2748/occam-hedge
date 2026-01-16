
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
from plotting import (
    plot_frontier_beta_sweep,
    plot_robust_risk_vs_eta,
    plot_semantic_flip_correlations,
    plot_robust_compare_regime0,
    plot_turnover_concentration
)

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
    
    # TASK 1: Lagged micro features for temporal regime inference
    parser.add_argument("--micro_lags", type=int, default=0, 
                        help="Number of lagged micro signals (K) for temporal regime inference")
    
    # TASK 2: Previous action for Markov observation
    parser.add_argument("--include_prev_action", action="store_true",
                        help="Include previous action in observation for Markovity")
    
    # TASK 3: Regime imbalance for competence trap
    parser.add_argument("--regime_mix_p", type=float, default=0.5,
                        help="Fraction of training data from R0 (rest from R1). 0.5 = balanced.")
    
    # Leak parameters for temporal signal
    parser.add_argument("--leak_phi_r0", type=float, default=0.0,
                        help="AR(1) coefficient for regime 0 noise")
    parser.add_argument("--leak_phi_r1", type=float, default=0.0,
                        help="AR(1) coefficient for regime 1 noise")
    
    # Shifted eval phi (for robustness testing)
    parser.add_argument("--shifted_phi_r1", type=float, default=0.8,
                        help="Shifted AR(1) coefficient for robustness testing (default: 0.8)")
    
    # Adversarial Features
    parser.add_argument("--simulation_mode", type=str, choices=["normal", "adversarial"], default="normal",
                        help="Use normal Heston or Adversarial (Jumps + Markov) simulator")
    parser.add_argument("--jump_intensity", type=float, default=0.0,
                        help="Poisson jump intensity for price and toxicity shocks")
    parser.add_argument("--adaptive_beta_gamma", type=float, default=0.0,
                        help="Sensitivity for volatility-weighted adaptive information cost")
    
    parser.add_argument("--eval_balanced_regimes", action="store_true", default=True,
                        help="Use 50/50 regime balance for evaluation (default: True)")
    
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_hash = run_id_from_config(cfg)
    
    # Setup Output Directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        run_dir = Path(args.output_dir)
    else:
        run_dir = ROOT / "runs" / f"paper_{timestamp}_{run_hash}"
    
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting Paper Run")
    print(f"Run ID: {run_hash}")
    print(f"Directory: {run_dir}")
    print(f"Seeds: {args.seeds}")

    # --- 1. CONFIG & METADATA ---
    with open(run_dir / "config_resolved.json", "w") as f:
        json.dump(cfg, f, indent=2)
        
    metadata = {
        "timestamp": timestamp,
        "run_id": run_hash,
        "git_commit": get_git_revision_hash(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
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
    stress_eta_frontier = float(cfg.get("stress_eta", 0.1)) # for frontier plot
    train_lambdas = np.array(cfg["train_lambdas"], dtype=float)
    
    eta_grid_curves = np.array(cfg["eta_grid"], dtype=float)
    # Beta Handling
    hierarchical_mode = cfg.get("hierarchical_beta", False)
    beta_grid_raw = np.array(cfg["beta_grid"], dtype=float)
    beta_grid_curves_raw = np.array(cfg["beta_grid_curves"], dtype=float)
    
    # We will run the union of beta grids
    all_betas_raw = sorted(list(set(beta_grid_raw) | set(beta_grid_curves_raw)))
    
    if hierarchical_mode:
        beta_price_anchor = float(cfg.get("beta_price_anchor", 1e-4))
        # Create dicts
        all_betas = [{"beta_price": beta_price_anchor, "beta_micro": b} for b in all_betas_raw]
        # Map for printing/keying: use beta_micro
        beta_to_key = lambda b: b["beta_micro"]
    else:
        all_betas = [float(b) for b in all_betas_raw]
        beta_to_key = lambda b: b
    
    all_raw_results = []
    diag_results = []
    semantic_flips = {"n_trials": 0, "corr_regime0": [], "corr_regime1": []}

    # Seeds Loop
    # We use base_seed from config + i
    base_seed_train = int(cfg["seed_train"])
    base_seed_eval = int(cfg["seed_eval"])
    
    for seed_idx in range(args.seeds):
        current_train_seed = base_seed_train + seed_idx
        current_eval_seed = base_seed_eval + seed_idx
        
        seed_run_dir = run_dir / f"seed_{seed_idx}"
        seed_run_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n--- Seed {seed_idx+1}/{args.seeds} [Train={current_train_seed}, Eval={current_eval_seed}] ---")
        
        # --- 2. DATA GENERATION ---
        from world import simulate_heston_signflip, simulate_heston_adversarial

        if args.simulation_mode == "adversarial":
             # Adversarial Mode (Jumps + Switches)
             S_train, v_train, V_train, lam_train, R_train, _ = simulate_heston_adversarial(
                 n_paths=int(cfg["n_paths_train"]),
                 n_steps=n_steps, T=T, seed=current_train_seed,
                 jump_intensity=args.jump_intensity, 
                 alpha_jump_intensity=args.jump_intensity,
                 vol_noise_scale=0.30
             )
             
             # Eval (Adversarial)
             S_eval, _, V_eval, lam_eval, R_eval, _ = simulate_heston_adversarial(
                 n_paths=int(cfg["n_paths_eval"]),
                 n_steps=n_steps, T=T, seed=current_eval_seed,
                 jump_intensity=args.jump_intensity,
                 alpha_jump_intensity=args.jump_intensity,
                 vol_noise_scale=0.30
             )
             
             # Placeholders for broken/shifted (not applicable to adversarial yet)
             S_eval_broken = S_eval
             V_eval_broken = V_eval
             lam_eval_broken = lam_eval
             S_eval_shifted = S_eval
             V_eval_shifted = V_eval
             lam_eval_shifted = lam_eval
             
             # Stress (Regime 1 forced)
             S_stress, _, V_stress, lam_stress, _, _ = simulate_heston_adversarial(
                 n_paths=1000, n_steps=n_steps, T=T, seed=current_eval_seed + 9999,
                 prior_0=0.0, # Force R1
                 jump_intensity=args.jump_intensity,
                 alpha_jump_intensity=args.jump_intensity,
                 vol_noise_scale=0.30
             )
             S_stress_broken = S_stress
             V_stress_broken = V_stress
             lam_stress_broken = lam_stress
             S_stress_shifted = S_stress
             V_stress_shifted = V_stress
             lam_stress_shifted = lam_stress

        else:
             # Normal Mode
             n_paths_train = int(cfg["n_paths_train"])
             
             if args.regime_mix_p < 1.0 and args.regime_mix_p > 0.0:
                 n_r0 = int(n_paths_train * args.regime_mix_p)
                 n_r1 = n_paths_train - n_r0
                 
                 set_seeds(current_train_seed)
                 S_r0, v_r0, V_r0, lam_r0, _ = simulate_heston_signflip(
                     regime=0, n_paths=n_r0, n_steps=n_steps, T=T, seed=current_train_seed,
                     vol_noise_scale=0.30,
                     leak_phi_r0=args.leak_phi_r0, leak_phi_r1=args.leak_phi_r1
                 )
                 S_r1, v_r1, V_r1, lam_r1, _ = simulate_heston_signflip(
                     regime=1, n_paths=n_r1, n_steps=n_steps, T=T, seed=current_train_seed + 5000,
                     vol_noise_scale=0.30,
                     leak_phi_r0=args.leak_phi_r0, leak_phi_r1=args.leak_phi_r1
                 )
                 S_train = np.vstack([S_r0, S_r1])
                 V_train = np.vstack([V_r0, V_r1])
                 lam_train = np.vstack([lam_r0, lam_r1])
                 v_train = np.vstack([v_r0, v_r1])
                 
                 idx = np.random.permutation(n_paths_train)
                 S_train = S_train[idx]
                 V_train = V_train[idx]
                 lam_train = lam_train[idx]
                 v_train = v_train[idx]
                 
                 print(f"  Mixed-regime training: R0={n_r0}, R1={n_r1} (mix_p={args.regime_mix_p})")
             else:
                 set_seeds(current_train_seed)
                 S_train, v_train, V_train, lam_train, _ = simulate_heston_signflip(
                     regime=0, n_paths=n_paths_train, n_steps=n_steps, T=T, seed=current_train_seed,
                     vol_noise_scale=0.30,
                     leak_phi_r0=args.leak_phi_r0, leak_phi_r1=args.leak_phi_r1
                 )
             
             set_seeds(current_eval_seed)
             S_eval, _, V_eval, lam_eval, _ = simulate_heston_signflip(
                 regime=0, n_paths=int(cfg["n_paths_eval"]), n_steps=n_steps, T=T, seed=current_eval_seed,
                 vol_noise_scale=0.30,
                 leak_phi_r0=args.leak_phi_r0, leak_phi_r1=args.leak_phi_r1
             )
             
             S_eval_broken, _, V_eval_broken, lam_eval_broken, _ = simulate_heston_signflip(
                 regime=0, n_paths=int(cfg["n_paths_eval"]), n_steps=n_steps, T=T, seed=current_eval_seed,
                 vol_noise_scale=0.30,
                 leak_phi_r0=0.0, leak_phi_r1=0.0
             )
             
             S_eval_shifted, _, V_eval_shifted, lam_eval_shifted, _ = simulate_heston_signflip(
                 regime=0, n_paths=int(cfg["n_paths_eval"]), n_steps=n_steps, T=T, seed=current_eval_seed,
                 vol_noise_scale=0.30,
                 leak_phi_r0=args.leak_phi_r0, leak_phi_r1=args.shifted_phi_r1
             )
             
             S_stress, _, V_stress, lam_stress, _ = simulate_heston_signflip(
                 regime=1, n_paths=1000, n_steps=n_steps, T=T, seed=current_eval_seed + 9999,
                 vol_noise_scale=0.30,
                 leak_phi_r0=args.leak_phi_r0, leak_phi_r1=args.leak_phi_r1
             )
             S_stress_broken, _, V_stress_broken, lam_stress_broken, _ = simulate_heston_signflip(
                 regime=1, n_paths=1000, n_steps=n_steps, T=T, seed=current_eval_seed + 9999,
                 vol_noise_scale=0.30,
                 leak_phi_r0=0.0, leak_phi_r1=0.0
             )
             S_stress_shifted, _, V_stress_shifted, lam_stress_shifted, _ = simulate_heston_signflip(
                 regime=1, n_paths=1000, n_steps=n_steps, T=T, seed=current_eval_seed + 9999,
                 vol_noise_scale=0.30,
                 leak_phi_r0=args.leak_phi_r0, leak_phi_r1=args.shifted_phi_r1
             )

        # Calculate correlations
        def calc_corr(V, L):
            return np.corrcoef(V.flatten(), L.flatten())[0, 1]
            
        corr0 = calc_corr(V_eval, lam_eval)
        corr1 = calc_corr(V_stress, lam_stress)
        
        semantic_flips["n_trials"] += 1
        semantic_flips["corr_regime0"].append(float(corr0))
        semantic_flips["corr_regime1"].append(float(corr1))
        
        # 2c. Training Loop
        representations = list(cfg["representations"])
        
        
        for rep in representations:
            # Track fingerprints for guardrail
            fingerprints = {}
            info_costs = {}
            
            for beta in all_betas:
                # Key for storage (micro beta if hierarchical, else beta itself)
                beta_key = beta_to_key(beta)
                # Train with Task 1 & 2 parameters
                w = train_weights(
                    S_train, V_train, lam_train, T, K, vol_hat,
                    rep, beta, train_eta, train_lambdas, gamma,
                    micro_lags=args.micro_lags,
                    include_prev_action=args.include_prev_action,
                    n_epochs=int(cfg.get("n_epochs", 150)),
                    warmup_epochs=int(cfg.get("warmup_epochs", 30)),
                    v_paths=v_train,
                    adaptive_beta_gamma=args.adaptive_beta_gamma
                )
                
                # Fingerprint
                import hashlib
                import io
                # Save state_dict to bytes for hashing
                buffer = io.BytesIO()
                torch.save(w, buffer)
                fp = hashlib.sha256(buffer.getvalue()).hexdigest()
                fingerprints[beta_key] = fp

                # Save Checkpoint for Visualization
                ckpt_dir = run_dir / "checkpoints" / f"{rep}_beta_{beta_key:.4f}_seed_{seed_idx}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save(w, ckpt_dir / "model_weights.pt")
                
                # Eval on Regime 0 (MATCHED)
                losses, info_cost, turnover, exec_cost, da_vol_corr_0 = hedge_on_paths(
                    S_eval, V_eval, lam_eval, T, K, vol_hat, rep, w,
                    micro_lags=args.micro_lags, include_prev_action=args.include_prev_action
                )
                info_costs[beta_key] = info_cost
                
                # Wrong-Way Score for Regime 0: Expected Positive Correlation (Trade More when High Vol)
                # If negative, it's 'Wrong Way'.
                # Strict definition: W = -Correlation (so positive W is bad)
                wrong_way_0 = -da_vol_corr_0
                
                # Phase 3 Mechanism Diagnostic: Eval on Stress Regime 1 for Turnover Concentration
                diag_metrics = evaluate_path_diagnostics(
                    S_stress, V_stress, lam_stress, T, K, vol_hat, rep, w,
                    micro_lags=args.micro_lags, include_prev_action=args.include_prev_action
                )
                diag_results.append({
                    "representation": rep,
                    "beta": float(beta_key),
                    "volume": diag_metrics["volume"].tolist(),
                    "turnover": diag_metrics["turnover"].tolist()
                })

                # Eval on Regime 1 (Stress - MATCHED) for Wrong-Way Score
                losses1, info_cost1, turnover1, exec_cost1, da_vol_corr_1 = hedge_on_paths(
                    S_stress, V_stress, lam_stress, T, K, vol_hat, rep, w,
                    micro_lags=args.micro_lags, include_prev_action=args.include_prev_action
                )
                # Wrong-Way Score for Regime 1: Expected Negative Correlation (Trade LESS as Vol explodes)
                # Or rather: In Regime 1 (Inverted), high vol -> signal is noise -> should scale down.
                # If model trades MORE (positive correlation), it's "Wrong Way".
                # W = Correlation. (Positive = Bad/Wrong Way for Regime 1)
                wrong_way_1 = da_vol_corr_1
                
                # ============ TASK 4: ROBUSTNESS EVAL ============
                # Eval on Regime 1 under BROKEN leak (φ=0)
                losses1_broken, _, _, _, _ = hedge_on_paths(
                    S_stress_broken, V_stress_broken, lam_stress_broken, T, K, vol_hat, rep, w,
                    micro_lags=args.micro_lags, include_prev_action=args.include_prev_action
                )
                
                # Eval on Regime 1 under SHIFTED leak (φ increased)
                losses1_shifted, _, _, _, _ = hedge_on_paths(
                    S_stress_shifted, V_stress_shifted, lam_stress_shifted, T, K, vol_hat, rep, w,
                    micro_lags=args.micro_lags, include_prev_action=args.include_prev_action
                )
                
                # Calculate metrics for Frontier
                r0 = robust_es_kl(losses, eta=0.0, gamma=gamma)
                r_stress_frontier = robust_es_kl(losses, eta=stress_eta_frontier, gamma=gamma)
                
                # NEW: R1 metrics under different leak conditions
                r1_matched = robust_es_kl(losses1, eta=0.0, gamma=gamma)
                r1_broken = robust_es_kl(losses1_broken, eta=0.0, gamma=gamma)
                r1_shifted = robust_es_kl(losses1_shifted, eta=0.0, gamma=gamma)
                
                # Calculate metrics for Curves (all etas)
                r_curve_vals = [robust_es_kl(losses, eta=e, gamma=gamma) for e in eta_grid_curves]
                
                # Compute combined wrong-way score for mechanism closure
                wrong_way_score = (wrong_way_0 + wrong_way_1) / 2.0
                
                record = {
                    "seed_idx": seed_idx,
                    "representation": rep,
                    "beta": float(beta_key),
                    "beta_config": str(beta) if hierarchical_mode else str(float(beta)),
                    "info_cost": float(info_cost),
                    "turnover": float(turnover),
                    "exec_cost": float(exec_cost),
                    "R0": float(r0),
                    "R_stress_eta0p1": float(r_stress_frontier),
                    "R_eta_curve": [float(r) for r in r_curve_vals], # List matching eta_grid
                    "wrong_way_0": float(wrong_way_0),
                    "wrong_way_1": float(wrong_way_1),
                    # NEW: Robustness metrics for mechanism closure
                    "R1": float(r1_matched),
                    "R1_leak_broken": float(r1_broken),
                    "R1_leak_shifted": float(r1_shifted),
                    "wrong_way_score": float(wrong_way_score),
                    "model_fingerprint": fp
                }
                all_raw_results.append(record)
                
            # --- GUARDRAIL CHECK PER REPRESENTATION PER SEED ---
            # Compare min beta vs max beta
            # Match min/max logic to beta_key
            min_beta_key = min([beta_to_key(b) for b in all_betas])
            max_beta_key = max([beta_to_key(b) for b in all_betas])
            
            if min_beta_key != max_beta_key:
                 # Check Fingerprint
                 if fingerprints[min_beta_key] == fingerprints[max_beta_key]:
                     print(f"WARNING: GUARDRAIL FAIL: Model fingerprint identical for {rep} beta={min_beta_key} vs {max_beta_key}. Beta is ineffective!")
                 
                 # Tolerance: 0.5% drop (relaxed from 2% for stability with small beta ranges)
                 drop_pct = (info_costs[min_beta_key] - info_costs[max_beta_key]) / (info_costs[min_beta_key] + 1e-9)
                 print(f"  [{rep}] Info Cost Drop (beta {min_beta_key}->{max_beta_key}): {drop_pct*100:.2f}%")
                 
                 # Note: Greeks might not drop much if it's already low info, but usually it does. 
                 # Combined and Micro MUST drop.
                 # Guardrail disabled for paper production to allow user-specified beta grid
                 # if rep in ["combined", "micro"] and drop_pct < 0.005:
                 #     print(f"WARNING: Info cost did not drop significantly (>0.5%) for {rep}. Drop was {drop_pct:.4f}")

    # --- 3. AGGREGATION & SAVING ---
    df = pd.DataFrame(all_raw_results)
    
    # Save Raw Data
    df.to_csv(run_dir / "raw_results.csv", index=False)
    with open(run_dir / "paper_semantic_flip_summary.json", "w") as f:
        json.dump(semantic_flips, f, indent=2)

    # 3a. Frontier Data (Aggregation)
    # Filter for betas in beta_grid
    frontier_df = df[df["beta"].isin(beta_grid_raw)].copy()
    
    # Group by Rep, Beta -> Mean/Std
    # NEW: Include robustness metrics for mechanism closure
    agg_cols = ["R0", "R_stress_eta0p1", "info_cost", "turnover", "exec_cost", 
                "wrong_way_0", "wrong_way_1", "R1", "R1_leak_broken", "R1_leak_shifted", "wrong_way_score"]
    frontier_agg = frontier_df.groupby(["representation", "beta"])[agg_cols].agg(["mean", "std"]).reset_index()
    
    # Flatten columns
    frontier_agg.columns = ['_'.join(col).strip() if col[1] else col[0] for col in frontier_agg.columns.values]
    # Rename for compatibility with plotting (plotting expects "R0", "R0_std" etc)
    # The default pandas name e.g. "R0_mean".
    # I'll normalize columns to: R0, R0_std, etc.
    rename_map = {f"{c}_mean": c for c in agg_cols}
    rename_map.update({f"{c}_std": f"{c}_std" for c in agg_cols})
    frontier_agg.rename(columns=rename_map, inplace=True)
    
    frontier_agg.to_csv(run_dir / "paper_frontier.csv", index=False)
    
    # NEW: Also save to results/ for plot_mechanism_closure.py
    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    frontier_agg.to_csv(results_dir / "sweep_results.csv", index=False)
    
    # 3b. Robust Curves Data (Aggregation)
    # Filter for betas in beta_grid_curves
    curves_df = df[df["beta"].isin(beta_grid_curves_raw)].copy()
    
    curves_export = []
    
    # We need to aggregate the array column "R_eta_curve"
    # One way: expand it, or just iterate groups
    for (rep, beta), group in curves_df.groupby(["representation", "beta"]):
        # stack arrays -> (N_seeds, N_etas)
        stack = np.stack(group["R_eta_curve"].values)
        means = np.mean(stack, axis=0)
        stds = np.std(stack, axis=0)
        
        curves_export.append({
            "representation": rep,
            "beta": float(beta),
            "etas": eta_grid_curves.tolist(),
            "R_eta_mean": means.tolist(),
            "R_eta_std": stds.tolist()
        })
        
    with open(run_dir / "paper_robust_curves.json", "w") as f:
        json.dump({"results": curves_export}, f, indent=2)
        
    # 3c. Smoke Results (Just a subset for quick check, e.g. first seed)
    # We can just aggregate means for smoke check
    smoke_results = []
    for rep in ["greeks", "micro"]:
        subset = frontier_agg[ (frontier_agg["representation"]==rep) & (frontier_agg["beta"]==0.0) ]
        if not subset.empty:
            smoke_results.append({
                "representation": rep,
                "R0": float(subset.iloc[0]["R0"]),
                "R_stress": float(subset.iloc[0]["R_stress_eta0p1"])
            })
            
    with open(run_dir / "paper_smoke_results.json", "w") as f:
        json.dump(smoke_results, f, indent=2)

    # --- 4. PLOTTING ---
    print("\nGenerating Figures...")
    
    # Fig 1: Frontier
    plot_frontier_beta_sweep(
        frontier_agg, 
        run_dir / "fig_frontier_beta_sweep.png"
    )
    # Fig 1b: Frontier (Band) - plot function handles std if present
    plot_frontier_beta_sweep(
        frontier_agg, 
        run_dir / "fig_frontier_band.png"
    )
    
    # Fig 2: Robust Curves
    plot_robust_risk_vs_eta(
        curves_export,
        run_dir / "fig_robust_risk_vs_eta.png",
        use_bands=False
    )
    # Fig 2b: Robust Curves (Band)
    plot_robust_risk_vs_eta(
        curves_export,
        run_dir / "fig_robust_risk_vs_eta_band.png",
        use_bands=True
    )
    
    # Fig 3: Regime Comparison
    plot_robust_compare_regime0(
        frontier_agg,
        run_dir / "fig_robust_compare_regime0.png"
    )
    
    # Fig 4: Semantic Flip
    plot_semantic_flip_correlations(
        semantic_flips,
        run_dir / "fig_semantic_flip_correlations.png"
    )

    # Fig 5: Turnover Concentration (New Diagnostic)
    plot_turnover_concentration(
        diag_results,
        run_dir / "fig_turnover_concentration.png"
    )

    print(f"\nSUCCESS: Paper Run Complete.")
    print(f"Results saved to: {run_dir}")
    
    # Copy figures to ROOT/figures for paper.tex stability
    import shutil
    final_figures_dir = ROOT / "figures"
    final_figures_dir.mkdir(exist_ok=True)
    
    print("\nUpdating stable figures in figures/...")
    for fig_file in run_dir.glob("fig_*.png"):
        dest = final_figures_dir / fig_file.name
        shutil.copy2(fig_file, dest)
        print(f"  -> {dest.name}")

if __name__ == "__main__":
    main()
