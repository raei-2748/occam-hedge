"""
Test suite for Task 1 & 2: Lagged Micro Features and Previous Action.
Verifies:
1. Feature dimension calculation with lags
2. Feature construction with V_history
3. Training with micro_lags enabled
4. Previous action observation
"""
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from features import get_feature_dim, occam_features_torch, micro_signal_torch


def test_feature_dim_with_lags():
    """Test that get_feature_dim correctly accounts for micro_lags."""
    print("Testing feature dimension with lags...")
    
    # Base dimensions (no lags)
    assert get_feature_dim("greeks") == 1
    assert get_feature_dim("micro") == 3
    assert get_feature_dim("combined") == 2
    
    # With lags (only affects micro/combined)
    assert get_feature_dim("greeks", micro_lags=4) == 1, "Greeks should not gain lags"
    assert get_feature_dim("micro", micro_lags=4) == 3 + 4, "Micro should gain 4 lag dims"
    assert get_feature_dim("combined", micro_lags=4) == 2 + 4, "Combined should gain 4 lag dims"
    
    # With prev_action
    assert get_feature_dim("combined", include_prev_action=True) == 3
    assert get_feature_dim("combined", include_prev_action=True, micro_lags=4) == 2 + 4 + 1
    
    print("  ✓ Feature dimensions correct with lags")


def test_feature_construction_with_history():
    """Test that occam_features_torch correctly appends V_history."""
    print("Testing feature construction with V_history...")
    
    batch_size = 32
    device = torch.device("cpu")
    
    S_t = torch.ones(batch_size) * 100.0
    tau_t = torch.ones(batch_size) * 0.1
    V_t = torch.ones(batch_size) * 1.0
    K = 100.0
    vol_hat = 0.2
    
    # Without lags - should be 2D for combined
    feats_no_lags = occam_features_torch("combined", S_t, tau_t, V_t, K, vol_hat)
    assert feats_no_lags.shape == (batch_size, 2), f"Expected (32, 2), got {feats_no_lags.shape}"
    
    # With lags but no history (zeros)
    feats_with_lags_no_hist = occam_features_torch(
        "combined", S_t, tau_t, V_t, K, vol_hat, 
        micro_lags=4, V_history=None
    )
    assert feats_with_lags_no_hist.shape == (batch_size, 6), f"Expected (32, 6), got {feats_with_lags_no_hist.shape}"
    # Last 4 dims should be zeros
    assert torch.allclose(feats_with_lags_no_hist[:, 2:], torch.zeros(batch_size, 4))
    
    # With actual V_history
    V_history = torch.randn(batch_size, 4)
    feats_with_hist = occam_features_torch(
        "combined", S_t, tau_t, V_t, K, vol_hat,
        micro_lags=4, V_history=V_history
    )
    assert feats_with_hist.shape == (batch_size, 6)
    assert torch.allclose(feats_with_hist[:, 2:], V_history)
    
    print("  ✓ Feature construction with V_history correct")


def test_feature_construction_with_prev_action():
    """Test that occam_features_torch correctly appends a_prev."""
    print("Testing feature construction with previous action...")
    
    batch_size = 32
    S_t = torch.ones(batch_size) * 100.0
    tau_t = torch.ones(batch_size) * 0.1
    V_t = torch.ones(batch_size) * 1.0
    K = 100.0
    vol_hat = 0.2
    
    # With prev_action
    a_prev = torch.randn(batch_size)
    feats = occam_features_torch(
        "combined", S_t, tau_t, V_t, K, vol_hat,
        include_prev_action=True, a_prev=a_prev
    )
    assert feats.shape == (batch_size, 3), f"Expected (32, 3), got {feats.shape}"
    assert torch.allclose(feats[:, 2], a_prev)
    
    # With both lags and prev_action
    V_history = torch.randn(batch_size, 4)
    feats_both = occam_features_torch(
        "combined", S_t, tau_t, V_t, K, vol_hat,
        include_prev_action=True, a_prev=a_prev,
        micro_lags=4, V_history=V_history
    )
    assert feats_both.shape == (batch_size, 7), f"Expected (32, 7), got {feats_both.shape}"
    # Order: [delta, micro, lag1, lag2, lag3, lag4, a_prev]
    assert torch.allclose(feats_both[:, 2:6], V_history)
    assert torch.allclose(feats_both[:, 6], a_prev)
    
    print("  ✓ Feature construction with previous action correct")


def test_backward_compatibility():
    """Test that default parameters produce identical behavior."""
    print("Testing backward compatibility...")
    
    batch_size = 32
    S_t = torch.ones(batch_size) * 100.0
    tau_t = torch.ones(batch_size) * 0.1
    V_t = torch.ones(batch_size) * 1.0
    K = 100.0
    vol_hat = 0.2
    
    # Default call (backward compatible)
    feats_default = occam_features_torch("combined", S_t, tau_t, V_t, K, vol_hat)
    
    # Explicit zeros
    feats_explicit = occam_features_torch(
        "combined", S_t, tau_t, V_t, K, vol_hat,
        micro_lags=0, include_prev_action=False
    )
    
    assert torch.allclose(feats_default, feats_explicit)
    assert get_feature_dim("combined") == get_feature_dim("combined", micro_lags=0, include_prev_action=False)
    
    print("  ✓ Backward compatibility verified")


def test_training_with_lags():
    """Smoke test: ensure training runs with micro_lags enabled."""
    print("Testing training with micro_lags...")
    
    from world import simulate_heston_signflip
    from experiment_occam import train_weights
    
    # Small data for quick test
    S, _, V, lam, _ = simulate_heston_signflip(
        regime=0, n_paths=50, n_steps=20, T=1.0, seed=42,
        leak_phi_r0=0.0, leak_phi_r1=0.6
    )
    
    # Train with lags
    weights = train_weights(
        S, V, lam, T=1.0, K=100.0, vol_hat=0.2,
        representation="combined",
        beta=0.01,
        train_eta=0.0,
        train_lambdas=[0.01],
        gamma=0.95,
        micro_lags=4,
        include_prev_action=True,
        n_epochs=10,  # Very short for smoke test
        warmup_epochs=5
    )
    
    # Verify we got weights back
    assert weights is not None
    assert len(weights) > 0
    
    print("  ✓ Training with micro_lags completed successfully")


if __name__ == "__main__":
    print("=" * 60)
    print("TASK 1 & 2: LAGGED FEATURES AND PREV ACTION TESTS")
    print("=" * 60)
    print()
    
    try:
        test_feature_dim_with_lags()
        print()
        test_feature_construction_with_history()
        print()
        test_feature_construction_with_prev_action()
        print()
        test_backward_compatibility()
        print()
        test_training_with_lags()
        print()
        
        print("=" * 60)
        print("✓ ALL LAGGED FEATURES TESTS PASSED!")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
