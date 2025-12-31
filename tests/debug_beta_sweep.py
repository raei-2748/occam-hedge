"""
Debug script: minimal reproduction of run_beta_sweep.py to see why results are identical
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from experiment_occam import hedge_on_paths, train_weights
from world import simulate_heston_signflip
from paper_config import Config
from utils import set_seeds


def main():
    cfg = Config.load()
    set_seeds(cfg["seed_train"])
    
    # Just test greeks with 3 β values
    betas = [0.0, 0.5, 1.0]
    
    # Generate data
    S_train, _, V_train, lam_train, _ = simulate_heston_signflip(
        regime=0, n_paths=int(cfg["n_paths_train"]), 
        n_steps=int(cfg["n_steps"]), T=float(cfg["T"]), 
        seed=int(cfg["seed_train"])
    )
    S_eval, _, V_eval, lam_eval, _ = simulate_heston_signflip(
        regime=0, n_paths=int(cfg["n_paths_eval"]), 
        n_steps=int(cfg["n_steps"]), T=float(cfg["T"]),
        seed=int(cfg["seed_eval"])
    )
    
    train_lambdas = np.array(cfg["train_lambdas"], dtype=float)
    
    print("="*60)
    print("TRAINING PHASE")
    print("="*60)
    
    trained = {}
    for beta in betas:
        print(f"\n→ Training greeks β={beta}")
        w = train_weights(
            S_train, V_train, lam_train,
            T=float(cfg["T"]),
            K=float(cfg["K"]),
            vol_hat=float(cfg["vol_hat"]),
            representation="greeks",
            beta=float(beta),
            train_eta=float(cfg["train_eta"]),
            train_lambdas=train_lambdas,
            gamma=float(cfg["gamma"]),
        )
        trained[beta] = w
        print(f"  Trained weights: {w}")
    
    print("\n" + "="*60)
    print("EVALUATION PHASE")
    print("="*60)
    
    for beta in betas:
        w = trained[beta]
        print(f"\n→ Evaluating β={beta}, weights={w}")
        
        losses, info_cost, turnover, exec_cost = hedge_on_paths(
            S_eval, V_eval, lam_eval,
            T=float(cfg["T"]),
            K=float(cfg["K"]),
            vol_hat=float(cfg["vol_hat"]),
            representation="greeks",
            weights=w,
        )
        
        print(f"  info_cost: {info_cost:.15f}")
        print(f"  turnover:  {turnover:.15f}")
        print(f"  exec_cost: {exec_cost:.15f}")
        print(f"  mean_loss: {np.mean(losses):.6f}")


if __name__ == "__main__":
    main()
