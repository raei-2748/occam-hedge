"""
Diagnostic script to audit Semantic Inversion.
Simulates Regime 0 and 1, computes optimal one-step trade aggressiveness,
and checks for the 'X' shape (slope inversion).
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch

# Add src to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from world import simulate_heston_signflip
from utils import set_seeds

def simple_optimal_trade(target_pos, current_pos, lam):
    """
    Optimal trade towards target in one step with quadratic cost 0.5 * lam * (da)^2
    Objective: min 0.5*(a - target)^2 + 0.5*lam*(a - current)^2
    Solution: a* = (target + lam * current) / (1 + lam)
    Trade da = a* - current = (target - current) / (1 + lam)
    Aggressiveness u = |da| = |target - current| / (1 + lam)
    """
    return np.abs(target_pos - current_pos) / (1.0 + lam)

def diagnose_inversion():
    print("="*60)
    print("DIAGNOSTIC: Semantic Inversion Audit (The 'X' Plot)")
    print("="*60)
    
    seed = 100
    set_seeds(seed)
    
    # Simulate both regimes
    n_paths = 1000
    print(" Simulating Regime 0...")
    _, _, V0, lam0, _ = simulate_heston_signflip(regime=0, n_paths=n_paths, seed=seed)
    
    print(" Simulating Regime 1...")
    _, _, V1, lam1, _ = simulate_heston_signflip(regime=1, n_paths=n_paths, seed=seed)
    
    # Flatten arrays
    V0_flat = V0.flatten()
    lam0_flat = lam0.flatten()
    
    V1_flat = V1.flatten()
    lam1_flat = lam1.flatten()
    
    # Assume a fixed "ur-signal" magnitude |target - current| = 1 for visualization
    # This isolates the effect of Volume->Lambda->Trade
    frictionless_trade_size = 1.0
    
    u0 = frictionless_trade_size / (1.0 + lam0_flat)
    u1 = frictionless_trade_size / (1.0 + lam1_flat)
    
    print("\n Computing correlations (Volume vs Aggressiveness)...")
    corr0 = np.corrcoef(np.log(V0_flat), u0)[0,1]
    corr1 = np.corrcoef(np.log(V1_flat), u1)[0,1]
    
    print(f" Regime 0 Corr(LogVol, Trade): {corr0:.4f} (Expected > 0)")
    print(f" Regime 1 Corr(LogVol, Trade): {corr1:.4f} (Expected < 0)")
    
    # Check for conflict
    is_conflict = (np.sign(corr0) != np.sign(corr1))
    print(f" Inversion Detected: {is_conflict}")
    
    if is_conflict:
        print(" ✅ SUCCESS: Semantic Inversion confirmed.")
    else:
        print(" ❌ FAILURE: No sign flip detected.")
        
    # Plotting
    print("\n Generating diagnostic plot...")
    plt.figure(figsize=(10, 6))
    
    # Scatter sample for visualization (don't plot millions of points)
    idx0 = np.random.choice(len(V0_flat), 2000, replace=False)
    idx1 = np.random.choice(len(V1_flat), 2000, replace=False)
    
    plt.scatter(np.log(V0_flat[idx0]), u0[idx0], alpha=0.3, label=f'Regime 0 (Corr={corr0:.2f})', color='blue', s=2)
    plt.scatter(np.log(V1_flat[idx1]), u1[idx1], alpha=0.3, label=f'Regime 1 (Corr={corr1:.2f})', color='red', s=2)
    
    # Trend lines (LOWESS or just polyfit)
    p0 = np.polyfit(np.log(V0_flat), u0, 1)
    p1 = np.polyfit(np.log(V1_flat), u1, 1)
    
    x_range = np.linspace(min(np.log(V0_flat).min(), np.log(V1_flat).min()), 
                          max(np.log(V0_flat).max(), np.log(V1_flat).max()), 100)
    
    plt.plot(x_range, np.polyval(p0, x_range), color='darkblue', linewidth=2, linestyle='--')
    plt.plot(x_range, np.polyval(p1, x_range), color='darkred', linewidth=2, linestyle='--')
    
    plt.xlabel("Log Volume (Proxy)")
    plt.ylabel("Optimal Trade Aggressiveness |u*|")
    plt.title("Semantic Inversion Diagnostic: The 'X' Shape")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_path = ROOT / "diagnostic_inversion.png"
    plt.savefig(out_path)
    print(f" Plot saved to: {out_path}")
    
    return is_conflict

if __name__ == "__main__":
    success = diagnose_inversion()
    # No exit code so user can inspect
