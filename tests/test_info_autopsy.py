"""
Test the Information Autopsy patch.
Verifies that info_components are correctly tracked and logged.
"""
import torch
import sys
from pathlib import Path

# Add src to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from experiment_occam import compute_hedging_losses_torch, train_model
from policies import FactorizedVariationalPolicy


def test_info_components_shape():
    """Test that info_components has correct shape for different representations."""
    device = torch.device("cpu")
    
    # Test parameters
    n_paths = 100
    n_steps = 10
    T = 1.0
    K = 100.0
    vol_hat = 0.2
    
    # Generate dummy data
    S = torch.randn(n_paths, n_steps + 1, device=device) * 10 + 100
    V = torch.abs(torch.randn(n_paths, n_steps + 1, device=device)) * 0.04 + 0.04
    lam = torch.ones(n_paths, n_steps, device=device) * 0.01
    
    # Test "combined" representation (2 features: delta, micro)
    model = FactorizedVariationalPolicy(input_dim=2, latent_dim_per_feature=2).to(device)
    losses, info_cost, info_components = compute_hedging_losses_torch(
        model, S, V, lam, T, K, vol_hat, representation="combined"
    )
    
    assert info_components.shape[0] == 2, f"Expected 2 components for 'combined', got {info_components.shape[0]}"
    assert torch.isclose(info_cost, torch.sum(info_components), atol=1e-5), "Total info_cost should equal sum of components"
    
    print("✓ Test passed: info_components shape is correct for 'combined' representation")
    print(f"  info_dim_0 (Delta): {info_components[0].item():.4f}")
    print(f"  info_dim_1 (Micro): {info_components[1].item():.4f}")
    print(f"  Total: {info_cost.item():.4f}")


def test_training_history_logging():
    """Test that training history correctly logs per-dimension info costs."""
    device = torch.device("cpu")
    
    # Test parameters
    n_paths = 50
    n_steps = 5
    T = 1.0
    K = 100.0
    vol_hat = 0.2
    
    # Generate dummy data
    S = torch.randn(n_paths, n_steps + 1, device=device) * 10 + 100
    V = torch.abs(torch.randn(n_paths, n_steps + 1, device=device)) * 0.04 + 0.04
    lam = torch.ones(n_paths, n_steps, device=device) * 0.01
    
    # Create model
    model = FactorizedVariationalPolicy(input_dim=2, latent_dim_per_feature=2).to(device)
    
    # Training config
    training_config = {
        "beta": 0.1,
        "train_eta": 0.95,
        "train_lambdas": [0.5, 0.5],
        "gamma": 0.95,
        "lr": 0.01,
        "n_epochs": 30,
        "warmup_epochs": 10
    }
    
    # Train
    result = train_model(
        model, S, V, lam, training_config, 
        representation="combined", T=T, K=K, vol_hat=vol_hat
    )
    
    # Check history
    history = result["history"]
    assert len(history) > 0, "History should not be empty"
    
    # Check first log entry
    first_entry = history[0]
    assert "info_total" in first_entry, "History should contain 'info_total'"
    assert "info_dim_0" in first_entry, "History should contain 'info_dim_0'"
    assert "info_dim_1" in first_entry, "History should contain 'info_dim_1'"
    
    print("✓ Test passed: Training history correctly logs per-dimension info costs")
    print(f"  Sample entry (epoch {first_entry['epoch']}):")
    print(f"    info_total: {first_entry['info_total']:.4f}")
    print(f"    info_dim_0: {first_entry['info_dim_0']:.4f}")
    print(f"    info_dim_1: {first_entry['info_dim_1']:.4f}")
    
    # Check last entry to see evolution
    last_entry = history[-1]
    print(f"  Final entry (epoch {last_entry['epoch']}):")
    print(f"    info_total: {last_entry['info_total']:.4f}")
    print(f"    info_dim_0: {last_entry['info_dim_0']:.4f}")
    print(f"    info_dim_1: {last_entry['info_dim_1']:.4f}")


if __name__ == "__main__":
    print("Testing Information Autopsy Implementation...\n")
    test_info_components_shape()
    print()
    test_training_history_logging()
    print("\n✓ All tests passed!")
