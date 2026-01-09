"""
Test suite for Task B: Leaky Simulator (AR(1) noise).
Verifies:
1. Backward compatibility: φ=0 matches original behavior
2. Variance matching: mean/std of micro proxy match across regimes
3. Autocorrelation differs when leak enabled
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from world import simulate_heston_signflip


def test_backward_compatibility():
    """Test that φ=0 produces identical behavior to original."""
    print("Testing backward compatibility (φ=0)...")
    
    n_paths = 1000
    n_steps = 50
    seed = 42
    
    # Run with explicit φ=0
    S0, _, V0, lam0, meta0 = simulate_heston_signflip(
        regime=0, n_paths=n_paths, n_steps=n_steps, T=1.0, seed=seed,
        leak_phi_r0=0.0, leak_phi_r1=0.0
    )
    
    # Run with default (φ should be 0)
    S0_default, _, V0_default, lam0_default, meta0_default = simulate_heston_signflip(
        regime=0, n_paths=n_paths, n_steps=n_steps, T=1.0, seed=seed
    )
    
    # Should be identical (same RNG seed, same path)
    assert np.allclose(V0, V0_default, rtol=1e-10), "φ=0 should match default behavior!"
    assert np.allclose(S0, S0_default, rtol=1e-10), "Price paths should be identical"
    
    print("  ✓ φ=0 behavior matches original (backward compatible)")


def test_variance_matching():
    """Test that mean/std of micro proxy match across regimes even with different φ."""
    print("Testing variance matching across regimes...")
    
    n_paths = 5000
    n_steps = 100
    seed = 42
    
    # Regime 0: i.i.d. noise (φ=0)
    _, _, V0, _, _ = simulate_heston_signflip(
        regime=0, n_paths=n_paths, n_steps=n_steps, T=1.0, seed=seed,
        leak_phi_r0=0.0, leak_phi_r1=0.6
    )
    
    # Regime 1: AR(1) noise (φ=0.6)
    _, _, V1, _, _ = simulate_heston_signflip(
        regime=1, n_paths=n_paths, n_steps=n_steps, T=1.0, seed=seed,
        leak_phi_r0=0.0, leak_phi_r1=0.6
    )
    
    mean_V0 = np.mean(V0)
    mean_V1 = np.mean(V1)
    std_V0 = np.std(V0)
    std_V1 = np.std(V1)
    
    print(f"  Regime 0: mean={mean_V0:.4f}, std={std_V0:.4f}")
    print(f"  Regime 1: mean={mean_V1:.4f}, std={std_V1:.4f}")
    
    # Mean and std should be approximately equal (within 20% tolerance due to finite sample)
    mean_diff = abs(mean_V0 - mean_V1) / max(mean_V0, mean_V1)
    std_diff = abs(std_V0 - std_V1) / max(std_V0, std_V1)
    
    print(f"  Mean difference: {mean_diff*100:.1f}%")
    print(f"  Std difference: {std_diff*100:.1f}%")
    
    assert mean_diff < 0.20, f"Mean differs too much: {mean_diff*100:.1f}%"
    assert std_diff < 0.30, f"Std differs too much: {std_diff*100:.1f}%"
    
    print("  ✓ Variance matching verified")


def test_autocorrelation_differs():
    """Test that autocorrelation differs when leak is enabled."""
    print("Testing autocorrelation difference...")
    
    n_paths = 2000
    n_steps = 100
    seed = 42
    
    def compute_lag1_autocorr(V):
        """Compute average lag-1 autocorrelation across paths."""
        # V is (n_paths, n_steps)
        V_t = V[:, :-1].flatten()
        V_t1 = V[:, 1:].flatten()
        
        # Demean
        V_t_dm = V_t - np.mean(V_t)
        V_t1_dm = V_t1 - np.mean(V_t1)
        
        # Correlation
        corr = np.sum(V_t_dm * V_t1_dm) / (np.sqrt(np.sum(V_t_dm**2)) * np.sqrt(np.sum(V_t1_dm**2)) + 1e-8)
        return corr
    
    # Regime 0: i.i.d. (φ=0)
    _, _, V0_iid, _, _ = simulate_heston_signflip(
        regime=0, n_paths=n_paths, n_steps=n_steps, T=1.0, seed=seed,
        leak_phi_r0=0.0, leak_phi_r1=0.6
    )
    
    # Regime 1: AR(1) (φ=0.6)
    _, _, V1_ar, _, _ = simulate_heston_signflip(
        regime=1, n_paths=n_paths, n_steps=n_steps, T=1.0, seed=seed,
        leak_phi_r0=0.0, leak_phi_r1=0.6
    )
    
    acf_r0 = compute_lag1_autocorr(V0_iid)
    acf_r1 = compute_lag1_autocorr(V1_ar)
    
    print(f"  Regime 0 (φ=0) ACF(1): {acf_r0:.4f}")
    print(f"  Regime 1 (φ=0.6) ACF(1): {acf_r1:.4f}")
    print(f"  Difference: {abs(acf_r1 - acf_r0):.4f}")
    
    # AR(1) should have higher autocorrelation
    # Note: The vol_proxy is exp(signal + noise), so ACF won't be exactly φ
    # But the AR(1) regime should have noticeably higher ACF
    assert acf_r1 > acf_r0 + 0.05, f"AR(1) regime should have higher ACF! Got R0={acf_r0:.4f}, R1={acf_r1:.4f}"
    
    print("  ✓ Autocorrelation differs as expected")


def test_meta_contains_leak_params():
    """Test that meta dict includes leak parameters."""
    print("Testing meta dict contains leak params...")
    
    _, _, _, _, meta = simulate_heston_signflip(
        regime=0, n_paths=10, n_steps=10, T=1.0, seed=42,
        leak_phi_r0=0.3, leak_phi_r1=0.7
    )
    
    assert "leak_phi_r0" in meta, "meta should contain leak_phi_r0"
    assert "leak_phi_r1" in meta, "meta should contain leak_phi_r1"
    assert meta["leak_phi_r0"] == 0.3, f"Expected 0.3, got {meta['leak_phi_r0']}"
    assert meta["leak_phi_r1"] == 0.7, f"Expected 0.7, got {meta['leak_phi_r1']}"
    
    print("  ✓ Meta dict contains leak parameters")


if __name__ == "__main__":
    print("=" * 60)
    print("TASK B: LEAKY SIMULATOR TESTS")
    print("=" * 60)
    print()
    
    try:
        test_backward_compatibility()
        print()
        test_variance_matching()
        print()
        test_autocorrelation_differs()
        print()
        test_meta_contains_leak_params()
        print()
        
        print("=" * 60)
        print("✓ ALL LEAKY SIMULATOR TESTS PASSED!")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
