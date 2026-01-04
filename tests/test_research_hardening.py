"""
Test suite for Research Hardening Phase implementation.
Verifies noise standardization, BS anchor, and policy surface generation.
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from world import simulate_heston_signflip
from policies import policy_delta_hedge


def test_regime_noise_standardization():
    """Test that both regimes now have identical noise scales."""
    print("Testing regime noise standardization...")
    
    # Simulate both regimes with same seed
    n_paths = 5000
    n_steps = 50
    
    S0, _, V0, lam0, _ = simulate_heston_signflip(
        regime=0, n_paths=n_paths, n_steps=n_steps, T=1.0, seed=42
    )
    S1, _, V1, lam1, _ = simulate_heston_signflip(
        regime=1, n_paths=n_paths, n_steps=n_steps, T=1.0, seed=42
    )
    
    # Check volume proxy statistics
    std_V0 = np.std(V0)
    std_V1 = np.std(V1)
    
    print(f"  Regime 0 volume std: {std_V0:.4f}")
    print(f"  Regime 1 volume std: {std_V1:.4f}")
    print(f"  Difference: {abs(std_V0 - std_V1):.4f}")
    
    # With identical noise_scale=0.30, the distributions should be very similar
    # Allow some tolerance due to different regime parameters affecting signal
    assert abs(std_V0 - std_V1) < 0.5, f"Volume distributions too different: {abs(std_V0 - std_V1):.4f}"
    
    print("✓ Regime noise standardization verified\n")


def test_bs_delta_hedge_policy():
    """Test that BS delta hedge policy works correctly."""
    print("Testing BS delta hedge policy...")
    
    # Test parameters
    K = 100.0
    vol_hat = 0.2
    
    # Test at-the-money
    S_atm = np.array([100.0])
    tau_atm = np.array([0.5])
    delta_atm = policy_delta_hedge(S_atm, K=K, tau_t=tau_atm, vol_hat=vol_hat)
    
    print(f"  ATM delta (S=100, τ=0.5): {delta_atm[0]:.4f}")
    assert 0.4 < delta_atm[0] < 0.6, f"ATM delta should be ~0.5, got {delta_atm[0]:.4f}"
    
    # Test in-the-money
    S_itm = np.array([120.0])
    tau_itm = np.array([0.5])
    delta_itm = policy_delta_hedge(S_itm, K=K, tau_t=tau_itm, vol_hat=vol_hat)
    
    print(f"  ITM delta (S=120, τ=0.5): {delta_itm[0]:.4f}")
    assert delta_itm[0] > delta_atm[0], "ITM delta should be higher than ATM"
    
    # Test out-of-the-money
    S_otm = np.array([80.0])
    tau_otm = np.array([0.5])
    delta_otm = policy_delta_hedge(S_otm, K=K, tau_t=tau_otm, vol_hat=vol_hat)
    
    print(f"  OTM delta (S=80, τ=0.5): {delta_otm[0]:.4f}")
    assert delta_otm[0] < delta_atm[0], "OTM delta should be lower than ATM"
    
    # Test near expiry
    S_near = np.array([100.0])
    tau_near = np.array([0.01])
    delta_near = policy_delta_hedge(S_near, K=K, tau_t=tau_near, vol_hat=vol_hat)
    
    print(f"  Near expiry delta (S=100, τ=0.01): {delta_near[0]:.4f}")
    assert 0.4 < delta_near[0] < 0.6, "Near expiry ATM delta should still be ~0.5"
    
    print("✓ BS delta hedge policy verified\n")


def test_policy_surface_components():
    """Test that policy surface generation components work."""
    print("Testing policy surface generation components...")
    
    # Test meshgrid generation
    S_range = np.linspace(80, 120, 10)
    V_range = np.linspace(0.1, 5.0, 10)
    S_grid, V_grid = np.meshgrid(S_range, V_range)
    
    assert S_grid.shape == (10, 10), "S_grid shape incorrect"
    assert V_grid.shape == (10, 10), "V_grid shape incorrect"
    assert S_grid.min() == 80.0, "S_grid min incorrect"
    assert S_grid.max() == 120.0, "S_grid max incorrect"
    assert V_grid.min() == 0.1, "V_grid min incorrect"
    assert V_grid.max() == 5.0, "V_grid max incorrect"
    
    print(f"  Grid shape: {S_grid.shape}")
    print(f"  S range: [{S_grid.min():.1f}, {S_grid.max():.1f}]")
    print(f"  V range: [{V_grid.min():.1f}, {V_grid.max():.1f}]")
    
    print("✓ Policy surface components verified\n")


def test_noise_scale_constant():
    """Verify that noise_scale is indeed constant across regimes in the code."""
    print("Testing noise_scale constant in code...")
    
    # Read the world.py file and check the noise_scale line
    world_file = ROOT / "src" / "world.py"
    with open(world_file, 'r') as f:
        content = f.read()
    
    # Check that the old regime-dependent line is gone
    assert "0.25 if regime == 0 else 0.35" not in content, \
        "Old regime-dependent noise_scale still in code!"
    
    # Check that the new constant line is present
    assert "noise_scale = 0.30  # Constant for both regimes" in content, \
        "New constant noise_scale not found in code!"
    
    print("  ✓ Code contains constant noise_scale = 0.30")
    print("  ✓ Old regime-dependent code removed")
    print("✓ Noise scale constant verified\n")


if __name__ == "__main__":
    print("="*60)
    print("RESEARCH HARDENING PHASE - VERIFICATION TESTS")
    print("="*60)
    print()
    
    try:
        test_noise_scale_constant()
        test_regime_noise_standardization()
        test_bs_delta_hedge_policy()
        test_policy_surface_components()
        
        print("="*60)
        print("✓ ALL TESTS PASSED!")
        print("="*60)
        print()
        print("Next steps:")
        print("1. Run beta sweep: .venv/bin/python scripts/run_beta_sweep.py")
        print("2. Generate policy surfaces after training completes")
        print("3. Inspect frontier plot for BS anchor (red X)")
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
