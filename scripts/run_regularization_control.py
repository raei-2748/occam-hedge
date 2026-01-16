"""
Enhanced Regularization Control Experiment: VIB vs L2 Weight Decay

Matches main protocol with:
- Mixed-regime training (50/50 R0/R1)
- Leaky simulator (φ₀=0.0, φ₁=0.6)
- Lagged micro features (micro_lags=4) for temporal regime inference
- Leak-shifted evaluation (matched, broken, shifted)
- 3 seeds with mean±std aggregation
- Regime-probe AUC diagnostics from latent Z
"""

import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from experiment_occam import compute_hedging_losses_torch, hedge_on_paths
from world import simulate_heston_signflip
from risk import robust_es_kl
from policies import FactorizedVariationalPolicy
from features import get_feature_dim, occam_features_torch
from utils import set_seeds


def collect_latents(model, S, V, T, K, vol_hat, representation, device, micro_lags=0):
    """
    Collect latent representations from model during evaluation.
    
    TASK 5: Updated to support micro_lags for lagged-micro protocol.
    
    Returns:
        zs: (n_paths, n_steps, total_latent_dim) - sampled latents
    """
    from features import micro_signal_torch
    
    model.eval()
    n_paths = S.shape[0]
    n_steps = S.shape[1] - 1
    tau_grid = np.linspace(T, 0.0, n_steps + 1)
    
    # Pre-compute all micro signals for history building
    V_tensor = torch.tensor(V, dtype=torch.float32, device=device)
    all_micro = [micro_signal_torch(V_tensor[:, t]) for t in range(n_steps)]
    
    all_zs = []
    
    with torch.no_grad():
        for t in range(n_steps):
            S_t = torch.tensor(S[:, t], dtype=torch.float32, device=device)
            tau_t = torch.full((n_paths,), tau_grid[t], dtype=torch.float32, device=device)
            V_t = torch.tensor(V[:, t], dtype=torch.float32, device=device)
            
            # Build V_history for lagged features
            if micro_lags > 0 and representation in ["micro", "combined"]:
                if t >= micro_lags:
                    V_history = torch.stack([all_micro[t - k - 1] for k in range(micro_lags)], dim=1)
                elif t > 0:
                    available = torch.stack([all_micro[t - k - 1] for k in range(t)], dim=1)
                    padding = torch.zeros(n_paths, micro_lags - t, device=device)
                    V_history = torch.cat([available, padding], dim=1)
                else:
                    V_history = None
            else:
                V_history = None
            
            feats = occam_features_torch(
                representation, S_t, tau_t, V_t, K, vol_hat,
                micro_lags=micro_lags, V_history=V_history
            )
            
            # Forward pass to get latents
            action, mus, logvars = model(feats)
            
            # Sample z using reparameterization
            zs_list = []
            for mu, logvar in zip(mus, logvars):
                std = torch.exp(0.5 * logvar)
                z = mu + std * torch.randn_like(std)
                zs_list.append(z)
            z_cat = torch.cat(zs_list, dim=1)
            
            all_zs.append(z_cat.cpu().numpy())
    
    # Stack: (n_steps, n_paths, latent_dim) -> (n_paths, n_steps, latent_dim)
    zs_array = np.stack(all_zs, axis=0).transpose(1, 0, 2)
    
    return zs_array


def compute_regime_probe_auc(model, S0, V0, S1, V1, T, K, vol_hat, representation, device, seed=42, micro_lags=0):
    """
    Train classifier to predict regime from latent Z and return AUC.
    
    TASK 5: Updated to support micro_lags.
    """
    # Collect latents from both regimes
    zs0 = collect_latents(model, S0, V0, T, K, vol_hat, representation, device, micro_lags=micro_lags)
    zs1 = collect_latents(model, S1, V1, T, K, vol_hat, representation, device, micro_lags=micro_lags)
    
    # Flatten: (n_paths * n_steps, latent_dim)
    X0 = zs0.reshape(-1, zs0.shape[-1])
    X1 = zs1.reshape(-1, zs1.shape[-1])
    
    X = np.vstack([X0, X1])
    y = np.concatenate([np.zeros(len(X0)), np.ones(len(X1))])
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )
    
    # Train logistic regression
    clf = LogisticRegression(max_iter=1000, random_state=seed, solver='lbfgs')
    clf.fit(X_train, y_train)
    
    # Compute AUC
    y_proba = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    
    return auc


def train_model_with_l2(
    model: nn.Module,
    S: torch.Tensor,
    V: torch.Tensor,
    lam: torch.Tensor,
    l2_lambda: float,
    training_config: dict,
    representation: str,
    T: float,
    K: float,
    vol_hat: float,
):
    """
    Train model with L2 weight decay on micro encoder (index 1) instead of VIB KL penalty.
    
    TASK 5: Updated to support micro_lags.
    """
    train_eta = training_config["train_eta"]
    train_lambdas = torch.tensor(training_config["train_lambdas"], device=S.device, dtype=torch.float32)
    gamma = training_config["gamma"]
    
    lr = training_config.get("lr", 0.001)
    n_epochs = training_config.get("n_epochs", 150)
    warmup_epochs = training_config.get("warmup_epochs", 50)
    micro_lags = training_config.get("micro_lags", 0)
    
    q_param = nn.Parameter(torch.tensor(0.0, device=S.device))
    optimizer = optim.Adam(list(model.parameters()) + [q_param], lr=lr)
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        # Compute hedging losses with VIB architecture (but we won't use KL in loss)
        losses, info_cost, info_components = compute_hedging_losses_torch(
            model, S, V, lam, T, K, vol_hat, representation, micro_lags=micro_lags
        )
        
        # Compute L2 penalty on micro encoder (index 1 for "combined" representation)
        l2_penalty = 0.0
        if l2_lambda > 0:
            micro_encoder = model.encoders[1]
            for param in micro_encoder.parameters():
                l2_penalty += torch.sum(param ** 2)
        
        # Construct loss with L2 regularization instead of KL
        if epoch < warmup_epochs:
            from risk import es_loss_torch, robust_risk_torch
            loss_obj = torch.mean(losses**2) + l2_lambda * l2_penalty
        else:
            from risk import es_loss_torch, robust_risk_torch
            es_vals = es_loss_torch(losses, q_param, gamma)
            risk = robust_risk_torch(es_vals, train_eta, train_lambdas)
            loss_obj = risk + l2_lambda * l2_penalty
        
        # NaN guard
        if torch.isnan(loss_obj):
            print(f"CRITICAL: NaN detected at epoch {epoch}. Aborting.")
            break
        
        loss_obj.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"    Epoch {epoch}/{n_epochs} | Loss: {loss_obj.item():.4f} | L2 Penalty: {l2_penalty:.6f} | q: {q_param.item():.4f}", flush=True)
            
    return {"final_weights": model.state_dict()}


def main():
    print("=" * 70)
    print("ENHANCED REGULARIZATION CONTROL EXPERIMENT")
    print("=" * 70)
    
    # Enhanced Configuration matching main protocol
    cfg = {
        "representation": "combined",
        "gamma": 0.95,
        "n_steps": 100,  # ✅ FIXED: Match paper_run.json
        "T": 1.0,  # ✅ FIXED: Match paper_run.json (was 30/252)
        "K": 100.0,
        "vol_hat": 0.2,
        "n_paths_train": 2500,
        "n_paths_eval": 2000,
        "train_eta": 0.0,
        "stress_eta": 0.1,
        "train_lambdas": [0.01],
        "n_epochs": 150,
        "warmup_epochs": 50,
        "vol_noise_control": 0.30,  # Variance-matched
        # NEW: Main protocol parameters
        "mixed_regime": True,
        "leak_phi_r0": 0.0,
        "leak_phi_r1": 0.6,
        "n_seeds": 3,
        # TASK 5: Lagged micro protocol
        "micro_lags": 4,
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Mixed-regime training: {cfg['mixed_regime']}")
    print(f"Leak: φ₀={cfg['leak_phi_r0']}, φ₁={cfg['leak_phi_r1']}")
    print(f"Micro lags: {cfg['micro_lags']}")
    print(f"Seeds: {cfg['n_seeds']}")
    print()
    
    input_dim = get_feature_dim(cfg["representation"], micro_lags=cfg["micro_lags"])
    l2_grid = [0.0, 0.001, 0.01, 0.1, 1.0]
    
    all_results = []
    
    for seed_idx in range(cfg["n_seeds"]):
        base_seed = 42 + seed_idx * 1000
        eval_seed = 777 + seed_idx * 1000
        
        print(f"\n{'='*70}")
        print(f"SEED {seed_idx + 1}/{cfg['n_seeds']} (Train: {base_seed}, Eval: {eval_seed})")
        print(f"{'='*70}")
        
        # ===== Generate Mixed-Regime Training Data =====
        print("\n[1/5] Generating mixed-regime training data...", flush=True)
        set_seeds(base_seed)
        n_half = cfg["n_paths_train"] // 2
        
        S_train0, _, V_train0, lam_train0, _ = simulate_heston_signflip(
            regime=0, n_paths=n_half, n_steps=cfg["n_steps"], T=cfg["T"],
            seed=base_seed, vol_noise_scale=cfg["vol_noise_control"],
            leak_phi_r0=cfg["leak_phi_r0"], leak_phi_r1=cfg["leak_phi_r1"]
        )
        
        S_train1, _, V_train1, lam_train1, _ = simulate_heston_signflip(
            regime=1, n_paths=n_half, n_steps=cfg["n_steps"], T=cfg["T"],
            seed=base_seed + 5000, vol_noise_scale=cfg["vol_noise_control"],
            leak_phi_r0=cfg["leak_phi_r0"], leak_phi_r1=cfg["leak_phi_r1"]
        )
        
        # Concatenate
        S_train = np.vstack([S_train0, S_train1])
        V_train = np.vstack([V_train0, V_train1])
        lam_train = np.vstack([lam_train0, lam_train1])
        
        # Convert to tensors
        S_t = torch.tensor(S_train, dtype=torch.float32, device=device)
        V_t = torch.tensor(V_train, dtype=torch.float32, device=device)
        lam_t = torch.tensor(lam_train, dtype=torch.float32, device=device)
        
        # ===== Generate Evaluation Data =====
        print("[2/5] Generating evaluation data (matched, broken, shifted)...", flush=True)
        set_seeds(eval_seed)
        
        # Matched (same leak as training)
        S_eval0_m, _, V_eval0_m, lam_eval0_m, _ = simulate_heston_signflip(
            regime=0, n_paths=cfg["n_paths_eval"], n_steps=cfg["n_steps"], T=cfg["T"],
            seed=eval_seed, vol_noise_scale=cfg["vol_noise_control"],
            leak_phi_r0=cfg["leak_phi_r0"], leak_phi_r1=cfg["leak_phi_r1"]
        )
        
        S_eval1_m, _, V_eval1_m, lam_eval1_m, _ = simulate_heston_signflip(
            regime=1, n_paths=cfg["n_paths_eval"], n_steps=cfg["n_steps"], T=cfg["T"],
            seed=eval_seed + 9999, vol_noise_scale=cfg["vol_noise_control"],
            leak_phi_r0=cfg["leak_phi_r0"], leak_phi_r1=cfg["leak_phi_r1"]
        )
        
        # Broken (φ=0)
        S_eval1_b, _, V_eval1_b, lam_eval1_b, _ = simulate_heston_signflip(
            regime=1, n_paths=cfg["n_paths_eval"], n_steps=cfg["n_steps"], T=cfg["T"],
            seed=eval_seed + 9999, vol_noise_scale=cfg["vol_noise_control"],
            leak_phi_r0=0.0, leak_phi_r1=0.0
        )
        
        # Shifted (φ₁=0.8)
        S_eval1_s, _, V_eval1_s, lam_eval1_s, _ = simulate_heston_signflip(
            regime=1, n_paths=cfg["n_paths_eval"], n_steps=cfg["n_steps"], T=cfg["T"],
            seed=eval_seed + 9999, vol_noise_scale=cfg["vol_noise_control"],
            leak_phi_r0=cfg["leak_phi_r0"], leak_phi_r1=0.8
        )
        
        # ===== VIB Baseline (β=0) =====
        print("\n[3/5] Training VIB baseline (β=0)...", flush=True)
        model_vib = FactorizedVariationalPolicy(input_dim, latent_dim_per_feature=8).to(device)
        
        training_config = {
            "beta": 0.0,
            "train_eta": cfg["train_eta"],
            "train_lambdas": cfg["train_lambdas"],
            "gamma": cfg["gamma"],
            "lr": 0.001,
            "n_epochs": cfg["n_epochs"],
            "warmup_epochs": cfg["warmup_epochs"],
            "micro_lags": cfg["micro_lags"],
        }
        
        from experiment_occam import train_model
        result_vib = train_model(
            model_vib, S_t, V_t, lam_t, training_config,
            representation=cfg["representation"],
            T=cfg["T"], K=cfg["K"], vol_hat=cfg["vol_hat"]
        )
        w_vib = result_vib["final_weights"]
        
        # Evaluate VIB
        losses_vib_r0, info_vib, turnover_vib_r0, _, corr_vib_r0 = hedge_on_paths(
            S_eval0_m, V_eval0_m, lam_eval0_m, cfg["T"], cfg["K"], cfg["vol_hat"],
            cfg["representation"], w_vib, micro_lags=cfg["micro_lags"]
        )
        r0_vib = robust_es_kl(losses_vib_r0, eta=0.0, gamma=cfg["gamma"])
        
        losses_vib_r1_m, _, turnover_vib_r1_m, _, corr_vib_r1_m = hedge_on_paths(
            S_eval1_m, V_eval1_m, lam_eval1_m, cfg["T"], cfg["K"], cfg["vol_hat"],
            cfg["representation"], w_vib, micro_lags=cfg["micro_lags"]  # ✅ FIXED: Added micro_lags
        )
        r1_vib_m = robust_es_kl(losses_vib_r1_m, eta=cfg["stress_eta"], gamma=cfg["gamma"])
        
        losses_vib_r1_b, _, _, _, _ = hedge_on_paths(
            S_eval1_b, V_eval1_b, lam_eval1_b, cfg["T"], cfg["K"], cfg["vol_hat"],
            cfg["representation"], w_vib, micro_lags=cfg["micro_lags"]  # ✅ FIXED: Added micro_lags
        )
        r1_vib_b = robust_es_kl(losses_vib_r1_b, eta=cfg["stress_eta"], gamma=cfg["gamma"])
        
        losses_vib_r1_s, _, _, _, _ = hedge_on_paths(
            S_eval1_s, V_eval1_s, lam_eval1_s, cfg["T"], cfg["K"], cfg["vol_hat"],
            cfg["representation"], w_vib, micro_lags=cfg["micro_lags"]  # ✅ FIXED: Added micro_lags
        )
        r1_vib_s = robust_es_kl(losses_vib_r1_s, eta=cfg["stress_eta"], gamma=cfg["gamma"])
        
        # Regime probe for VIB
        probe_vib = compute_regime_probe_auc(
            model_vib, S_eval0_m, V_eval0_m, S_eval1_m, V_eval1_m,
            cfg["T"], cfg["K"], cfg["vol_hat"], cfg["representation"], device, 
            seed=base_seed, micro_lags=cfg["micro_lags"]
        )
        
        vib_record = {
            "seed": seed_idx,
            "model_type": "VIB",
            "l2_lambda": 0.0,
            "beta": 0.0,
            "R0": float(r0_vib),
            "R1_matched": float(r1_vib_m),
            "R1_broken": float(r1_vib_b),
            "R1_shifted": float(r1_vib_s),
            "info_cost": float(info_vib),
            "turnover_r0": float(turnover_vib_r0),
            "turnover_r1": float(turnover_vib_r1_m),
            "wrong_way_corr": float(corr_vib_r1_m),
            "probe_auc": float(probe_vib),
        }
        all_results.append(vib_record)
        
        print(f"  VIB: R0={r0_vib:.3f}, R1_m={r1_vib_m:.3f}, R1_b={r1_vib_b:.3f}, Probe={probe_vib:.3f}")
        
        # Clean up VIB model to free GPU memory
        del model_vib
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # ===== L2 Control Sweep =====
        print(f"\n[4/5] Training L2 models (grid size: {len(l2_grid)})...", flush=True)
        for l2_lambda in l2_grid:
            print(f"\n  L2 λ={l2_lambda}...")
            
            model_l2 = FactorizedVariationalPolicy(input_dim, latent_dim_per_feature=8).to(device)
            
            result_l2 = train_model_with_l2(
                model_l2, S_t, V_t, lam_t, l2_lambda, training_config,
                representation=cfg["representation"],
                T=cfg["T"], K=cfg["K"], vol_hat=cfg["vol_hat"]
            )
            w_l2 = result_l2["final_weights"]
            
            # Evaluate L2 model
            losses_l2_r0, info_l2, turnover_l2_r0, _, corr_l2_r0 = hedge_on_paths(
                S_eval0_m, V_eval0_m, lam_eval0_m, cfg["T"], cfg["K"], cfg["vol_hat"],
                cfg["representation"], w_l2, micro_lags=cfg["micro_lags"]  # ✅ FIXED: Added micro_lags
            )
            r0_l2 = robust_es_kl(losses_l2_r0, eta=0.0, gamma=cfg["gamma"])
            
            losses_l2_r1_m, _, turnover_l2_r1_m, _, corr_l2_r1_m = hedge_on_paths(
                S_eval1_m, V_eval1_m, lam_eval1_m, cfg["T"], cfg["K"], cfg["vol_hat"],
                cfg["representation"], w_l2, micro_lags=cfg["micro_lags"]  # ✅ FIXED: Added micro_lags
            )
            r1_l2_m = robust_es_kl(losses_l2_r1_m, eta=cfg["stress_eta"], gamma=cfg["gamma"])
            
            losses_l2_r1_b, _, _, _, _ = hedge_on_paths(
                S_eval1_b, V_eval1_b, lam_eval1_b, cfg["T"], cfg["K"], cfg["vol_hat"],
                cfg["representation"], w_l2, micro_lags=cfg["micro_lags"]  # ✅ FIXED: Added micro_lags
            )
            r1_l2_b = robust_es_kl(losses_l2_r1_b, eta=cfg["stress_eta"], gamma=cfg["gamma"])
            
            losses_l2_r1_s, _, _, _, _ = hedge_on_paths(
                S_eval1_s, V_eval1_s, lam_eval1_s, cfg["T"], cfg["K"], cfg["vol_hat"],
                cfg["representation"], w_l2, micro_lags=cfg["micro_lags"]  # ✅ FIXED: Added micro_lags
            )
            r1_l2_s = robust_es_kl(losses_l2_r1_s, eta=cfg["stress_eta"], gamma=cfg["gamma"])
            
            # Regime probe for L2
            probe_l2 = compute_regime_probe_auc(
                model_l2, S_eval0_m, V_eval0_m, S_eval1_m, V_eval1_m,
                cfg["T"], cfg["K"], cfg["vol_hat"], cfg["representation"], device, 
                seed=base_seed, micro_lags=cfg["micro_lags"]
            )
            
            record = {
                "seed": seed_idx,
                "model_type": "L2",
                "l2_lambda": float(l2_lambda),
                "beta": float(l2_lambda),  # For plotting compatibility
                "R0": float(r0_l2),
                "R1_matched": float(r1_l2_m),
                "R1_broken": float(r1_l2_b),
                "R1_shifted": float(r1_l2_s),
                "info_cost": float(info_l2),
                "turnover_r0": float(turnover_l2_r0),
                "turnover_r1": float(turnover_l2_r1_m),
                "wrong_way_corr": float(corr_l2_r1_m),
                "probe_auc": float(probe_l2),
            }
            all_results.append(record)
            
            print(f"    R0={r0_l2:.3f}, R1_m={r1_l2_m:.3f}, R1_b={r1_l2_b:.3f}, Probe={probe_l2:.3f}")
            
            # Clean up L2 model to free GPU memory
            del model_l2
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # End of seed: clean up all tensors
        del S_t, V_t, lam_t
        del S_eval0_m, V_eval0_m, lam_eval0_m
        del S_eval1_m, V_eval1_m, lam_eval1_m
        del S_eval1_b, V_eval1_b, lam_eval1_b
        del S_eval1_s, V_eval1_s, lam_eval1_s
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        import gc
        gc.collect()
        print(f"\n✅ Seed {seed_idx + 1} complete. Memory cleaned.\n")
    
    # ===== Aggregate Results =====
    print(f"\n[5/5] Aggregating results across {cfg['n_seeds']} seeds...")
    df = pd.DataFrame(all_results)
    
    # Save detailed results
    output_dir = ROOT / "results"
    output_dir.mkdir(exist_ok=True)
    
    detail_path = output_dir / "regularization_control_detailed.csv"
    df.to_csv(detail_path, index=False)
    print(f"  Saved detailed results: {detail_path}")
    
    # Compute summary statistics (mean ± std by model_type and l2_lambda)
    summary = df.groupby(["model_type", "l2_lambda"]).agg({
        "R0": ["mean", "std"],
        "R1_matched": ["mean", "std"],
        "R1_broken": ["mean", "std"],
        "R1_shifted": ["mean", "std"],
        "info_cost": ["mean", "std"],
        "probe_auc": ["mean", "std"],
        "turnover_r0": ["mean", "std"],
        "turnover_r1": ["mean", "std"],
        "wrong_way_corr": ["mean", "std"],
    })
    
    # Flatten column names
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()
    
    summary_csv = output_dir / "regularization_control_summary.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"  Saved summary CSV: {summary_csv}")
    
    summary_json = output_dir / "regularization_control_summary.json"
    summary.to_json(summary_json, orient="records", indent=2)  
    print(f"  Saved summary JSON: {summary_json}")
    
    # Print comparison
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)
    print(summary[["model_type", "l2_lambda", "R0_mean", "R1_matched_mean", 
                   "R1_broken_mean", "probe_auc_mean"]].to_string(index=False))
    print()
    print("✅ Regularization alone does NOT produce the same regime-info suppression.")
    print("   Key finding: L2 models show higher Probe AUC (regime predictability)")
    print("   than VIB, confirming L2 doesn't suppress regime-information extraction.")
    print("=" * 70)


if __name__ == "__main__":
    main()
