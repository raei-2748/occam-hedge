"""
Diagnostic: Check if run_robustness_curves.py actually trains distinct models per β
"""
import sys
from pathlib import Path
import numpy as np
import hashlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from experiment_occam import train_weights
from world import simulate_heston_signflip
from utils import set_seeds
from paper_config import load_config


def weights_fingerprint(w):
    """Compute stable hash of weights"""
    return hashlib.sha256(w.tobytes()).hexdigest()[:12]


def main():
    from paper_config import Config
    cfg = Config.load()
    set_seeds(cfg["seed_train"])
    
    # Use actual config from run_robustness_curves.py
    betas = np.array(cfg["beta_grid_curves"], dtype=float)
    representations = ["greeks"]  # Just one for speed
    
    n_steps = int(cfg["n_steps"])
    T = float(cfg["T"])
    train_lambdas = np.array(cfg["train_lambdas"], dtype=float)
    
    # Generate training data ONCE (like the script does)
    S_train, _, V_train, lam_train, _ = simulate_heston_signflip(
        regime=0, n_paths=int(cfg["n_paths_train"]), 
        n_steps=n_steps, T=T, seed=int(cfg["seed_train"])
    )
    
    print("Training models for each β...")
    trained = {}
    for rep in representations:
        for beta in betas:
            print(f"  Training {rep} β={beta}")
            w = train_weights(
                S_train, V_train, lam_train,
                T=T, K=float(cfg["K"]), vol_hat=float(cfg["vol_hat"]),
                representation=rep,
                beta=float(beta),
                train_eta=float(cfg["train_eta"]),
                train_lambdas=train_lambdas,
                gamma=float(cfg["gamma"]),
            )
            trained[(rep, float(beta))] = w
            fp = weights_fingerprint(w)
            print(f"    → Fingerprint: {fp}, weights: {w}")
    
    # Check if all fingerprints are the same
    fps = [weights_fingerprint(w) for w in trained.values()]
    unique_fps = set(fps)
    
    print(f"\n{'='*60}")
    print(f"Trained {len(betas)} models")
    print(f"Unique fingerprints: {len(unique_fps)}")
    print(f"Fingerprints: {fps}")
    
    if len(unique_fps) == 1:
        print("\n❌ FAILURE MODE B: All models are IDENTICAL!")
        print("   → β is not affecting training OR models are being reused")
        return False
    else:
        print("\n✓ Models differ across β")
        return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
