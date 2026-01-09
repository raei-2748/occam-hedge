"""
Regression test for the loss function sign.
Ensures trading costs INCREASE the computed loss (not decrease it).
"""
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from experiment_occam import compute_hedging_losses_torch
from policies import FactorizedVariationalPolicy
from features import get_feature_dim


def test_cost_increases_loss():
    """
    Verify that trading costs increase the computed loss.
    
    In a scenario where payoff == pnl (perfect hedge), the only source of loss 
    should be transaction costs. With the correct formula:
        loss = payoff - pnl + cost = 0 + cost > 0
    
    With the buggy formula (- cost), loss would be negative.
    """
    device = torch.device("cpu")
    
    # Minimal simulation setup
    n_paths = 100
    n_steps = 10
    T = 1.0
    K = 100.0
    vol_hat = 0.2
    representation = "greeks"
    
    # Create constant price paths (S never changes -> payoff = 0 for K=100)
    S = torch.full((n_paths, n_steps + 1), 100.0, device=device)
    V = torch.ones((n_paths, n_steps), device=device)
    
    # HIGH impact coefficient to ensure measurable cost
    lam = torch.ones((n_paths, n_steps), device=device) * 0.5
    
    # Initialize a random model
    input_dim = get_feature_dim(representation)
    model = FactorizedVariationalPolicy(input_dim, latent_dim_per_feature=8).to(device)
    model.eval()
    
    with torch.no_grad():
        losses, info_cost, _ = compute_hedging_losses_torch(
            model, S, V, lam, T, K, vol_hat, representation
        )
    
    # Since S is constant at K=100, payoff = max(100 - 100, 0) = 0
    # Since S never changes, pnl = sum(a * dS) = 0
    # Therefore: loss = payoff - pnl + cost = 0 - 0 + cost = cost
    
    mean_loss = losses.mean().item()
    
    # With the correct formula, loss should be POSITIVE (equal to cost)
    # With the buggy formula, loss would be NEGATIVE
    assert mean_loss >= 0.0, (
        f"Loss should be non-negative when costs are the only source. "
        f"Got mean_loss={mean_loss:.6f}. This suggests costs are being subtracted!"
    )
    
    print(f"✓ Mean loss with constant paths (cost only): {mean_loss:.6f}")
    print("✓ Costs correctly increase the loss (positive sign)")


def test_cost_sign_direction():
    """
    Verify that increasing the impact coefficient INCREASES the loss.
    """
    device = torch.device("cpu")
    
    n_paths = 200
    n_steps = 20
    T = 1.0
    K = 100.0
    vol_hat = 0.2
    representation = "greeks"
    
    # Simple random paths
    torch.manual_seed(42)
    S = 100.0 + torch.cumsum(torch.randn(n_paths, n_steps + 1) * 0.5, dim=1)
    S[:, 0] = 100.0
    V = torch.ones((n_paths, n_steps), device=device)
    
    input_dim = get_feature_dim(representation)
    model = FactorizedVariationalPolicy(input_dim, latent_dim_per_feature=8).to(device)
    model.eval()
    
    # Test with LOW impact
    lam_low = torch.ones((n_paths, n_steps), device=device) * 0.01
    with torch.no_grad():
        losses_low, _, _ = compute_hedging_losses_torch(
            model, S, V, lam_low, T, K, vol_hat, representation
        )
    
    # Test with HIGH impact
    lam_high = torch.ones((n_paths, n_steps), device=device) * 1.0
    with torch.no_grad():
        losses_high, _, _ = compute_hedging_losses_torch(
            model, S, V, lam_high, T, K, vol_hat, representation
        )
    
    mean_low = losses_low.mean().item()
    mean_high = losses_high.mean().item()
    
    print(f"  Low impact (λ=0.01): mean loss = {mean_low:.6f}")
    print(f"  High impact (λ=1.0): mean loss = {mean_high:.6f}")
    
    # Higher costs should mean higher loss
    assert mean_high > mean_low, (
        f"Higher impact should produce higher loss. "
        f"Got low={mean_low:.6f}, high={mean_high:.6f}"
    )
    
    print("✓ Higher trading costs correctly increase the loss")


if __name__ == "__main__":
    print("=" * 60)
    print("LOSS FUNCTION SIGN VERIFICATION")
    print("=" * 60)
    print()
    
    test_cost_increases_loss()
    print()
    test_cost_sign_direction()
    
    print()
    print("=" * 60)
    print("✓ ALL LOSS SIGN TESTS PASSED")
    print("=" * 60)
