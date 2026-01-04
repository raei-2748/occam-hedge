"""
Diagnostic test to verify β has an effect on training.
This must fail if β is not working.
"""
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from experiment_occam import train_weights
from world import simulate_heston_signflip
from utils import set_seeds


def test_beta_effect():
    """Test that β=0 and β=100 produce different models"""
    set_seeds(42)
    
    # Minimal training data
    S, _, V, lam, _ = simulate_heston_signflip(
        regime=0, n_paths=500, n_steps=50, T=1.0, seed=42
    )
    
    train_lambdas = np.logspace(-2, 1, 20)
    
    # Train with β=0
    set_seeds(42)  # Reset to ensure same initialization
    w_beta0 = train_weights(
        S, V, lam,
        T=1.0, K=100.0, vol_hat=0.2,
        representation="greeks",
        beta=0.0,
        train_eta=0.05,
        train_lambdas=train_lambdas,
        gamma=0.95
    )
    
    # Train with β=100
    set_seeds(42)  # Reset to ensure same initialization
    w_beta100 = train_weights(
        S, V, lam,
        T=1.0, K=100.0, vol_hat=0.2,
        representation="greeks",
        beta=100.0,
        train_eta=0.05,
        train_lambdas=train_lambdas,
        gamma=0.95
    )
    
    # Compute difference
    diff = 0.0
    for k in w_beta0.keys():
        diff += torch.norm(w_beta0[k] - w_beta100[k]).item()
    
    # print(f"β=0 weights keys: {list(w_beta0.keys())}")
    print(f"L2 difference across all parameters: {diff}")
    
    # CRITICAL: If β is working, weights MUST differ
    assert diff > 0.001, f"β has no effect! Weights are identical (diff={diff})"
    
    print("✓ β penalty is working - weights differ significantly")


if __name__ == "__main__":
    test_beta_effect()
