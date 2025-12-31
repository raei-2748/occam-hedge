"""
GUARDRAIL TEST: Verify β penalty has measurable effect

This test ensures that increasing the information penalty β actually
changes the trained policy. If this test fails, β is not connected
to the training objective.

Expected behavior:
- Higher β should reduce info_cost
- Higher β should reduce turnover (policy becomes simpler)
- Weights should differ across β values
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from experiment_occam import train_weights, hedge_on_paths
from world import simulate_heston_signflip
from utils import set_seeds


def test_beta_penalty_effect():
    """
    Test that β has a measurable effect on trained policies.
    Uses extreme β values (0 vs 100) to make the effect obvious.
    """
    print("="*60)
    print("GUARDRAIL TEST: β Penalty Effect")
    print("="*60)
    
    # Fixed seed for reproducibility
    seed = 12345
    set_seeds(seed)
    
    # Minimal training data for speed
    S_train, _, V_train, lam_train, _ = simulate_heston_signflip(
        regime=0, n_paths=1000, n_steps=50, T=1.0, seed=seed
    )
    S_eval, _, V_eval, lam_eval, _ = simulate_heston_signflip(
        regime=0, n_paths=2000, n_steps=50, T=1.0, seed=seed + 999
    )
    
    train_lambdas = np.logspace(-2, 1, 30)
    
    # Train with extreme β values
    results = {}
    for beta in [0.0, 100.0]:
        print(f"\n→ Training with β={beta}")
        set_seeds(seed)  # Reset for fair comparison
        
        w = train_weights(
            S_train, V_train, lam_train,
            T=1.0, K=100.0, vol_hat=0.2,
            representation="greeks",
            beta=beta,
            train_eta=0.05,
            train_lambdas=train_lambdas,
            gamma=0.95
        )
        
        # Evaluate
        losses, info_cost, turnover, exec_cost = hedge_on_paths(
            S_eval, V_eval, lam_eval,
            T=1.0, K=100.0, vol_hat=0.2,
            representation="greeks",
            weights=w
        )
        
        results[beta] = {
            'weights': w,
            'info_cost': info_cost,
            'turnover': turnover,
            'exec_cost': exec_cost
        }
        
        print(f"  Weights: {w}")
        print(f"  Info cost: {info_cost:.6f}")
        print(f"  Turnover: {turnover:.6f}")
    
    # Verification
    print("\n" + "="*60)
    print("VERIFICATION")
    print("="*60)
    
    w0 = results[0.0]['weights']
    w100 = results[100.0]['weights']
    info0 = results[0.0]['info_cost']
    info100 = results[100.0]['info_cost']
    
    weight_diff = np.linalg.norm(w0 - w100)
    info_reduction = (info0 - info100) / info0 if info0 > 0 else 0
    
    print(f"Weight difference (L2): {weight_diff:.6f}")
    print(f"Info cost reduction: {info_reduction*100:.2f}%")
    
    # CRITICAL ASSERTIONS
    errors = []
    
    if weight_diff < 0.01:
        errors.append(f"Weights are too similar (diff={weight_diff:.6f})")
    
    if info_reduction < 0.05:  # Expect at least 5% reduction
        errors.append(f"Info cost barely changed (reduction={info_reduction*100:.2f}%)")
    
    if errors:
        print("\n❌ GUARDRAIL FAILURE:")
        for err in errors:
            print(f"   - {err}")
        print("\n→ β penalty is NOT affecting training!")
        return False
    
    print("\n✓ β penalty is working correctly")
    print(f"  → Weights differ by {weight_diff:.4f}")
    print(f"  → Info cost reduced by {info_reduction*100:.1f}%")
    return True


if __name__ == "__main__":
    success = test_beta_penalty_effect()
    sys.exit(0 if success else 1)
