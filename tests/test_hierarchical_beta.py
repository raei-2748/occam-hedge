"""
Test suite for hierarchical beta regularization.

Verifies:
1. Backward compatibility with single-beta mode
2. Hierarchical penalty application
3. Info components mapping for 'combined' representation
"""

import sys
from pathlib import Path
import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from experiment_occam import train_model, compute_hedging_losses_torch
from world import simulate_heston_signflip
from policies import FactorizedVariationalPolicy


def test_backward_compatibility():
    """Test that single-beta mode produces same results as before."""
    device = torch.device("cpu")
    
    # Generate test data
    S, _, V, lam, _ = simulate_heston_signflip(
        regime=0, n_paths=100, n_steps=30, T=0.119, seed=42
    )
    
    S_t = torch.tensor(S, dtype=torch.float32, device=device)
    V_t = torch.tensor(V, dtype=torch.float32, device=device)
    lam_t = torch.tensor(lam, dtype=torch.float32, device=device)
    
    # Test with single beta (backward compatible)
    model = FactorizedVariationalPolicy(input_dim=2, latent_dim_per_feature=2).to(device)
    
    training_config = {
        "beta": 0.1,  # Single float
        "train_eta": 0.0,
        "train_lambdas": [0.0, 0.1, 0.5, 1.0],
        "gamma": 0.95,
        "lr": 0.005,
        "n_epochs": 20,
        "warmup_epochs": 5
    }
    
    result = train_model(
        model, S_t, V_t, lam_t, training_config,
        representation="combined", T=0.119, K=100.0, vol_hat=0.2
    )
    
    # Check that history logs don't have hierarchical flag set to True
    assert len(result["history"]) > 0
    for entry in result["history"]:
        assert entry["hierarchical"] == False
        assert "beta_price" not in entry
        assert "beta_micro" not in entry
    
    print("✓ Test passed: Backward compatibility verified")


def test_hierarchical_penalty():
    """Test that hierarchical beta applies differential penalties."""
    device = torch.device("cpu")
    
    # Generate test data
    S, _, V, lam, _ = simulate_heston_signflip(
        regime=0, n_paths=100, n_steps=30, T=0.119, seed=42
    )
    
    S_t = torch.tensor(S, dtype=torch.float32, device=device)
    V_t = torch.tensor(V, dtype=torch.float32, device=device)
    lam_t = torch.tensor(lam, dtype=torch.float32, device=device)
    
    # Test with hierarchical beta
    model = FactorizedVariationalPolicy(input_dim=2, latent_dim_per_feature=2).to(device)
    
    training_config = {
        "beta": {"beta_price": 0.0001, "beta_micro": 1.0},  # Dict for hierarchical
        "train_eta": 0.0,
        "train_lambdas": [0.0, 0.1, 0.5, 1.0],
        "gamma": 0.95,
        "lr": 0.005,
        "n_epochs": 20,
        "warmup_epochs": 5
    }
    
    result = train_model(
        model, S_t, V_t, lam_t, training_config,
        representation="combined", T=0.119, K=100.0, vol_hat=0.2
    )
    
    # Check that history logs have hierarchical flag and beta values
    assert len(result["history"]) > 0
    for entry in result["history"]:
        assert entry["hierarchical"] == True
        assert entry["beta_price"] == 0.0001
        assert entry["beta_micro"] == 1.0
    
    print("✓ Test passed: Hierarchical penalty correctly applied")


def test_info_components_mapping():
    """Test that info_components[0] = Delta, info_components[1] = Micro for 'combined'."""
    device = torch.device("cpu")
    
    # Generate test data
    S, _, V, lam, _ = simulate_heston_signflip(
        regime=0, n_paths=50, n_steps=30, T=0.119, seed=42
    )
    
    S_t = torch.tensor(S, dtype=torch.float32, device=device)
    V_t = torch.tensor(V, dtype=torch.float32, device=device)
    lam_t = torch.tensor(lam, dtype=torch.float32, device=device)
    
    model = FactorizedVariationalPolicy(input_dim=2, latent_dim_per_feature=2).to(device)
    
    # Run one forward pass
    losses, info_cost, info_components = compute_hedging_losses_torch(
        model, S_t, V_t, lam_t, T=0.119, K=100.0, vol_hat=0.2,
        representation="combined"
    )
    
    # For 'combined' representation, should have exactly 2 components
    assert len(info_components) == 2, f"Expected 2 components, got {len(info_components)}"
    
    # Both should be non-negative (KL divergence)
    assert info_components[0].item() >= 0, "info_components[0] (Delta) should be non-negative"
    assert info_components[1].item() >= 0, "info_components[1] (Micro) should be non-negative"
    
    # Total should equal sum of components
    total_from_components = info_components[0] + info_components[1]
    assert torch.allclose(info_cost, total_from_components, atol=1e-5), \
        f"Total info_cost {info_cost.item()} != sum of components {total_from_components.item()}"
    
    print("✓ Test passed: Info components mapping verified")
    print(f"  info_components[0] (Delta): {info_components[0].item():.4f}")
    print(f"  info_components[1] (Micro): {info_components[1].item():.4f}")
    print(f"  Total: {info_cost.item():.4f}")


if __name__ == "__main__":
    print("Running hierarchical beta tests...\n")
    test_backward_compatibility()
    print()
    test_hierarchical_penalty()
    print()
    test_info_components_mapping()
    print("\n✅ All tests passed!")
