
"""
Validation Suite for Occam's Hedge.
Consolidates Oracle Test and Smoke Tests to ensure codebase integrity.
"""
import sys
import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from world import simulate_heston_signflip
from experiment_occam import train_weights, hedge_on_paths
from features import occam_features_torch
from risk import robust_es_kl, expected_shortfall
from utils import set_seeds
from paper_config import load_config, run_id_from_config
from policies import FactorizedVariationalPolicy

def run_oracle_test():
    print("\n" + "="*60)
    print("VALIDATION: Oracle Disambiguation Test")
    print("="*60)
    set_seeds(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    n_paths_per_regime = 5000
    T = 1.0
    n_steps = 100
    K = 100.0
    vol_hat = 0.2
    
    print(" Generating mixed regime data...")
    S0, _, V0, lam0, _ = simulate_heston_signflip(regime=0, n_paths=n_paths_per_regime, n_steps=n_steps)
    S1, _, V1, lam1, _ = simulate_heston_signflip(regime=1, n_paths=n_paths_per_regime, n_steps=n_steps)
    
    S = np.concatenate([S0, S1], axis=0)
    V = np.concatenate([V0, V1], axis=0)
    lam = np.concatenate([lam0, lam1], axis=0)
    
    # Regime labels
    R = np.concatenate([np.zeros((n_paths_per_regime, n_steps)), np.ones((n_paths_per_regime, n_steps))], axis=0)
    
    print(f" Dataset shape: {S.shape}")
    
    # Training Parameters
    beta = 0.0 
    train_eta = 0.05
    train_lambdas = [1.0] 
    gamma = 0.05
    
    print(" Training Oracle Policy (beta=0.0, 100 epochs)...")
    weights = train_weights(
        S, V, lam, T, K, vol_hat, 
        representation="oracle", 
        beta=beta, 
        train_eta=train_eta, 
        train_lambdas=train_lambdas, 
        gamma=gamma,
        R=torch.tensor(R, dtype=torch.float32, device=device),
        n_epochs=100
    )
    
    print(" Evaluating...")
    losses, info, _, _ = hedge_on_paths(
        S, V, lam, T, K, vol_hat, 
        representation="oracle", 
        weights_or_state_dict=weights,
        R=torch.tensor(R, dtype=torch.float32, device=device)
    )
    
    es_95 = expected_shortfall(losses, gamma=0.95)
    print(f" Oracle Aggregate ES_0.95: {es_95:.4f}")
    
    success = es_95 < 2.0
    print(f" Result: {'✅ SUCCESS' if success else '❌ FAILURE'}")
    return success

def run_competence_smoke():
    print("\n" + "="*60)
    print("VALIDATION: Competence Trap Smoke Test")
    print("="*60)
    seed = 42
    set_seeds(seed)
    
    print(" Generating data (Regime 0)...")
    S_train, _, V_train, lam_train, _ = simulate_heston_signflip(regime=0, n_paths=1000, n_steps=30, T=30/252, seed=seed)
    S_val, _, V_val, lam_val, _ = simulate_heston_signflip(regime=0, n_paths=1000, n_steps=30, T=30/252, seed=seed+1)
    
    K = 100.0
    vol_hat = 0.2
    T = 30/252
    
    print(" Training Greeks-Only...")
    w_g = train_weights(S_train, V_train, lam_train, T=T, K=K, vol_hat=vol_hat, representation="greeks", beta=0.0, train_eta=0.0, train_lambdas=[0.01], gamma=0.99, n_epochs=20)
    
    print(" Training Microstucture-Only...")
    w_m = train_weights(S_train, V_train, lam_train, T=T, K=K, vol_hat=vol_hat, representation="micro", beta=0.0, train_eta=0.0, train_lambdas=[0.01], gamma=0.99, n_epochs=50)
    
    es_g = expected_shortfall(hedge_on_paths(S_val, V_val, lam_val, T, K, vol_hat, "greeks", w_g)[0], gamma=0.95)
    es_m = expected_shortfall(hedge_on_paths(S_val, V_val, lam_val, T, K, vol_hat, "micro", w_m)[0], gamma=0.95)
    
    print(f" Greeks ES_95: {es_g:.4f}")
    print(f" Micro  ES_95: {es_m:.4f}")
    
    gap = es_g - es_m
    success = gap > 0.05
    print(f" Improvement: {gap:.4f}")
    print(f" Result: {'✅ SUCCESS' if success else '❌ FAILURE'}")
    return success

def run_empirical_smoke(config_path):
    print("\n" + "="*60)
    print(f"VALIDATION: Empirical Smoke Test ({Path(config_path).name})")
    print("="*60)
    cfg = load_config(config_path)
    run_id = run_id_from_config(cfg)
    smoke_cfg = cfg.get("smoke", {})
    if not smoke_cfg:
        print(" [WARNING] No 'smoke' section in config. Skipping.")
        return True
        
    set_seeds(int(smoke_cfg["seed"]))
    
    print(f" Running smoke for {run_id}...")
    # Simplified version of run_empirical_smoke_test.py core
    S_train, _, V_train, lam_train, _ = simulate_heston_signflip(
        regime=0, n_paths=int(smoke_cfg["n_paths_train"]), n_steps=int(cfg["n_steps"]), T=float(cfg["T"]), seed=int(smoke_cfg["seed"])
    )
    
    # Train combined policy if configured
    print(" Training Combined Policy...")
    w = train_weights(
        S_train, V_train, lam_train,
        T=float(cfg["T"]), K=float(cfg["K"]), vol_hat=float(cfg["vol_hat"]),
        representation="combined",
        beta=float(smoke_cfg.get("beta_combined", 0.1)),
        train_eta=float(cfg["train_eta"]),
        train_lambdas=np.array(cfg["train_lambdas"], dtype=float),
        gamma=float(cfg["gamma"]),
        n_epochs=30
    )
    
    print("✅ SUCCESS: Empirical Smoke Test completed.")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["all", "oracle", "competence", "empirical"], default="all")
    parser.add_argument("--config", default=str(ROOT / "configs" / "paper_run.json"))
    args = parser.parse_args()
    
    success = True
    if args.mode in ["all", "oracle"]:
        success &= run_oracle_test()
    if args.mode in ["all", "competence"]:
        success &= run_competence_smoke()
    if args.mode in ["all", "empirical"]:
        success &= run_empirical_smoke(args.config)
        
    sys.exit(0 if success else 1)
