#!/usr/bin/env python3
"""
TASK A: Identifiability Audit Script

Probes whether the current (variance-matched) simulator leaks regime information.
Trains classifiers to predict regime R from observations:
- Snapshot probe: R ← o_t (single timestep)
- Window probes: R ← o_{t−k:t} (k-step history)

Expected results:
- Without leak (φ=0): snapshot AUC ≈ 0.5, all probes ≈ chance
- With leak (φ>0): window probes > chance (temporal signal)
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from world import simulate_heston_signflip
from features import occam_features_torch, get_feature_dim


def extract_observations(S, V, T, K, vol_hat, representation="combined", micro_lags=0):
    """
    Extract the actual observation tensor the policy receives.
    Returns (n_paths * n_steps, feature_dim) array.
    
    TASK 4: Updated to support micro_lags for temporal regime inference testing.
    """
    from features import micro_signal_torch
    
    n_paths, n_steps_plus_1 = S.shape
    n_steps = n_steps_plus_1 - 1
    
    device = torch.device("cpu")
    tau_grid = np.linspace(T, 0.0, n_steps + 1)
    
    # Pre-compute all micro signals for history building
    V_tensor = torch.tensor(V, dtype=torch.float32, device=device)
    all_micro = [micro_signal_torch(V_tensor[:, t]) for t in range(n_steps)]
    
    all_obs = []
    for t in range(n_steps):
        S_t = torch.tensor(S[:, t], dtype=torch.float32, device=device)
        tau_t = torch.full((n_paths,), tau_grid[t], dtype=torch.float32, device=device)
        V_t = torch.tensor(V[:, t], dtype=torch.float32, device=device)
        
        # Build V_history for lagged features
        if micro_lags > 0 and representation in ["micro", "combined"]:
            if t >= micro_lags:
                V_history = torch.stack([all_micro[t - k - 1] for k in range(micro_lags)], dim=1)
            elif t > 0:
                available = torch.stack([all_micro[t - k - 1] for k in range(t)], dim=1)
                padding = torch.zeros(n_paths, micro_lags - t, device=device)
                V_history = torch.cat([available, padding], dim=1)
            else:
                V_history = None
        else:
            V_history = None
        
        feats = occam_features_torch(
            representation, S_t, tau_t, V_t, K, vol_hat,
            micro_lags=micro_lags, V_history=V_history
        )
        all_obs.append(feats.numpy())
    
    # Stack: (n_steps, n_paths, feature_dim) -> (n_paths, n_steps, feature_dim)
    obs_array = np.stack(all_obs, axis=0).transpose(1, 0, 2)
    return obs_array  # (n_paths, n_steps, feature_dim)


def build_snapshot_dataset(obs_r0, obs_r1):
    """
    Flatten all timesteps into single observations.
    X: (N, feature_dim), y: (N,)
    """
    n0, T0, D = obs_r0.shape
    n1, T1, D = obs_r1.shape
    
    X0 = obs_r0.reshape(-1, D)
    X1 = obs_r1.reshape(-1, D)
    
    X = np.vstack([X0, X1])
    y = np.concatenate([np.zeros(len(X0)), np.ones(len(X1))])
    
    return X, y


def build_window_dataset(obs_r0, obs_r1, window_size):
    """
    Build dataset with k-step history windows.
    X: (N, window_size * feature_dim), y: (N,)
    """
    def extract_windows(obs, k):
        n_paths, n_steps, D = obs.shape
        windows = []
        for t in range(k, n_steps):
            # (n_paths, k, D) -> (n_paths, k*D)
            window = obs[:, t-k:t, :].reshape(n_paths, -1)
            windows.append(window)
        return np.vstack(windows)  # (n_paths * (n_steps - k), k*D)
    
    X0 = extract_windows(obs_r0, window_size)
    X1 = extract_windows(obs_r1, window_size)
    
    X = np.vstack([X0, X1])
    y = np.concatenate([np.zeros(len(X0)), np.ones(len(X1))])
    
    return X, y


def train_probe(X, y, seed=42):
    """
    Train logistic regression probe with train/test split.
    Returns test AUC.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )
    
    clf = LogisticRegression(max_iter=1000, random_state=seed, solver='lbfgs')
    clf.fit(X_train, y_train)
    
    y_proba = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    
    return auc


def run_audit(n_episodes=200, horizon=30, T=30/252, K=100.0, vol_hat=0.2,
              representation="combined", seed=42,
              leak_phi_r0=0.0, leak_phi_r1=0.0, vol_noise_scale=0.30,
              micro_lags=0):
    """
    Main audit function. Generates data, trains probes, returns results.
    
    TASK 4: Updated to support micro_lags.
    """
    print("=" * 60)
    print("IDENTIFIABILITY AUDIT")
    print("=" * 60)
    print(f"Episodes per regime: {n_episodes}")
    print(f"Horizon: {horizon}")
    print(f"Representation: {representation}")
    print(f"Leak φ₀: {leak_phi_r0}, φ₁: {leak_phi_r1}")
    print()
    
    # Generate data for both regimes
    print("Generating Regime 0 data...")
    S0, _, V0, _, _ = simulate_heston_signflip(
        regime=0, n_paths=n_episodes, n_steps=horizon, T=T, seed=seed,
        vol_noise_scale=vol_noise_scale,
        leak_phi_r0=leak_phi_r0, leak_phi_r1=leak_phi_r1
    )
    
    print("Generating Regime 1 data...")
    S1, _, V1, _, _ = simulate_heston_signflip(
        regime=1, n_paths=n_episodes, n_steps=horizon, T=T, seed=seed + 9999,
        vol_noise_scale=vol_noise_scale,
        leak_phi_r0=leak_phi_r0, leak_phi_r1=leak_phi_r1
    )
    
    # Extract observations
    print("Extracting observations...")
    print(f"  Using micro_lags={micro_lags}")
    obs_r0 = extract_observations(S0, V0, T, K, vol_hat, representation, micro_lags=micro_lags)
    obs_r1 = extract_observations(S1, V1, T, K, vol_hat, representation, micro_lags=micro_lags)
    
    # Train probes
    results = []
    window_sizes = [1, 4, 16, 64]  # 1 = snapshot
    
    print("\nTraining probes...")
    print("-" * 60)
    print(f"{'Probe':<20} {'Window k':<12} {'AUC':<10}")
    print("-" * 60)
    
    for k in window_sizes:
        if k == 1:
            X, y = build_snapshot_dataset(obs_r0, obs_r1)
            probe_name = "snapshot"
        else:
            if k > horizon:
                print(f"  Skipping k={k} (larger than horizon={horizon})")
                continue
            X, y = build_window_dataset(obs_r0, obs_r1, k)
            probe_name = f"window_{k}"
        
        auc = train_probe(X, y, seed=seed)
        
        print(f"{probe_name:<20} {k:<12} {auc:.4f}")
        
        results.append({
            "representation": representation,
            "k": k,
            "probe": probe_name,
            "metric": "AUC",
            "value": float(auc),
            "seed": seed,
            "n_episodes": n_episodes,
            "horizon": horizon,
            "leak_phi_r0": leak_phi_r0,
            "leak_phi_r1": leak_phi_r1
        })
    
    print("-" * 60)
    
    # Summary interpretation
    snapshot_auc = results[0]["value"] if results else 0.5
    max_window_auc = max(r["value"] for r in results if r["k"] > 1) if len(results) > 1 else 0.5
    
    print("\n📊 INTERPRETATION:")
    if abs(snapshot_auc - 0.5) < 0.05:
        print("  ✓ Snapshot probe ≈ 0.5: No single-step regime leak")
    else:
        print(f"  ⚠ Snapshot probe = {snapshot_auc:.3f}: Marginal distribution differs!")
    
    if max_window_auc > 0.55:
        print(f"  ✓ Window probe AUC = {max_window_auc:.3f}: Temporal signal detected")
    else:
        print(f"  ⚠ Window probes ≈ 0.5: No temporal signal (regime unidentifiable)")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Identifiability Audit for VIB Experiments")
    parser.add_argument("--n_episodes", type=int, default=200, help="Episodes per regime")
    parser.add_argument("--horizon", type=int, default=30, help="Steps per episode")
    parser.add_argument("--representation", type=str, default="combined", 
                        choices=["greeks", "micro", "combined", "oracle"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--leak_phi_r0", type=float, default=0.0, help="AR(1) φ for regime 0")
    parser.add_argument("--leak_phi_r1", type=float, default=0.0, help="AR(1) φ for regime 1")
    parser.add_argument("--micro_lags", type=int, default=0, help="Lagged micro signals for temporal inference")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()
    
    results = run_audit(
        n_episodes=args.n_episodes,
        horizon=args.horizon,
        representation=args.representation,
        seed=args.seed,
        leak_phi_r0=args.leak_phi_r0,
        leak_phi_r1=args.leak_phi_r1,
        micro_lags=args.micro_lags
    )
    
    # Save results
    output_path = Path(args.output) if args.output else ROOT / "results" / "audit_identifiability.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_path}")


if __name__ == "__main__":
    main()
