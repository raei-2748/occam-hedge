#!/usr/bin/env python3
"""
TASK D: Regime Probe from Latent Z

Collects latent encoder outputs (z, μ) during evaluation and trains
classifiers to predict regime R from the latent representation.

Key acceptance test: 
- Regime predictability from Z decreases monotonically with β

Outputs: results/regime_probe_vs_beta.json
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from world import simulate_heston_signflip
from features import occam_features_torch, get_feature_dim
from policies import FactorizedVariationalPolicy
from experiment_occam import train_model


def collect_latents(model, S, V, T, K, vol_hat, representation, device, micro_lags=0):
    """
    Run model in eval mode and collect latent representations.
    
    TASK 4: Updated to support micro_lags for temporal regime inference testing.
    
    Returns:
        mus: (n_paths, n_steps, n_features * latent_dim) - mean vectors
        zs: (n_paths, n_steps, n_features * latent_dim) - sampled latents
    """
    from features import micro_signal_torch
    
    model.eval()
    n_paths = S.shape[0]
    n_steps = S.shape[1] - 1
    tau_grid = np.linspace(T, 0.0, n_steps + 1)
    
    # Pre-compute all micro signals for history building
    V_tensor = torch.tensor(V, dtype=torch.float32, device=device)
    all_micro = [micro_signal_torch(V_tensor[:, t]) for t in range(n_steps)]
    
    all_mus = []
    all_zs = []
    
    with torch.no_grad():
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
            
            # Forward pass to get latents
            action, mus, logvars = model(feats)
            
            # Concatenate mus from all encoders
            mu_cat = torch.cat(mus, dim=1)  # (n_paths, total_latent_dim)
            
            # Sample z using reparameterization
            zs_list = []
            for mu, logvar in zip(mus, logvars):
                std = torch.exp(0.5 * logvar)
                z = mu + std * torch.randn_like(std)
                zs_list.append(z)
            z_cat = torch.cat(zs_list, dim=1)
            
            all_mus.append(mu_cat.cpu().numpy())
            all_zs.append(z_cat.cpu().numpy())
    
    # Stack: (n_steps, n_paths, latent_dim) -> (n_paths, n_steps, latent_dim)
    mus_array = np.stack(all_mus, axis=0).transpose(1, 0, 2)
    zs_array = np.stack(all_zs, axis=0).transpose(1, 0, 2)
    
    return mus_array, zs_array


def train_regime_probe(Z0, Z1, window_size=1, seed=42):
    """
    Train classifier to predict regime from latent(s).
    
    Args:
        Z0: (n_paths, n_steps, latent_dim) - latents from regime 0
        Z1: (n_paths, n_steps, latent_dim) - latents from regime 1
        window_size: number of timesteps to use (1 = snapshot)
        
    Returns:
        AUC score
    """
    def extract_windows(Z, k):
        n_paths, n_steps, D = Z.shape
        windows = []
        for t in range(k, n_steps):
            window = Z[:, t-k:t, :].reshape(n_paths, -1)
            windows.append(window)
        return np.vstack(windows)
    
    if window_size == 1:
        X0 = Z0.reshape(-1, Z0.shape[-1])
        X1 = Z1.reshape(-1, Z1.shape[-1])
    else:
        X0 = extract_windows(Z0, window_size)
        X1 = extract_windows(Z1, window_size)
    
    X = np.vstack([X0, X1])
    y = np.concatenate([np.zeros(len(X0)), np.ones(len(X1))])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )
    
    clf = LogisticRegression(max_iter=1000, random_state=seed, solver='lbfgs')
    clf.fit(X_train, y_train)
    
    y_proba = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    
    return auc


def run_probe_sweep(betas, n_episodes=100, horizon=30, T=30/252, K=100.0, vol_hat=0.2,
                    representation="combined", seed=42, n_epochs=100,
                    leak_phi_r0=0.0, leak_phi_r1=0.0, micro_lags=0):
    """
    Train models at various beta values and probe regime predictability from Z.
    
    TASK 4: Updated to support micro_lags for temporal regime inference testing.
    """
    print("=" * 60)
    print("REGIME PROBE FROM LATENT Z")
    print("=" * 60)
    print(f"Betas: {betas}")
    print(f"Episodes: {n_episodes}, Horizon: {horizon}")
    print(f"Leak: φ₀={leak_phi_r0}, φ₁={leak_phi_r1}, micro_lags={micro_lags}")
    print()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate evaluation data for both regimes
    print("Generating evaluation data...")
    S0, _, V0, lam0, _ = simulate_heston_signflip(
        regime=0, n_paths=n_episodes, n_steps=horizon, T=T, seed=seed + 7777,
        leak_phi_r0=leak_phi_r0, leak_phi_r1=leak_phi_r1
    )
    S1, _, V1, lam1, _ = simulate_heston_signflip(
        regime=1, n_paths=n_episodes, n_steps=horizon, T=T, seed=seed + 8888,
        leak_phi_r0=leak_phi_r0, leak_phi_r1=leak_phi_r1
    )
    
    # Generate training data
    print("Generating training data...")
    S_train, _, V_train, lam_train, _ = simulate_heston_signflip(
        regime=0, n_paths=n_episodes * 3, n_steps=horizon, T=T, seed=seed,
        leak_phi_r0=leak_phi_r0, leak_phi_r1=leak_phi_r1
    )
    
    results = []
    
    print("\n" + "-" * 60)
    print(f"{'Beta':<12} {'Probe_Z':<12} {'Probe_μ':<12}")
    print("-" * 60)
    
    for beta in betas:
        # Create and train model with correct input dim for micro_lags
        input_dim = get_feature_dim(representation, micro_lags=micro_lags)
        model = FactorizedVariationalPolicy(input_dim, latent_dim_per_feature=2).to(device)
        
        training_config = {
            "beta": beta,
            "train_eta": 0.0,
            "train_lambdas": [0.01],
            "gamma": 0.95,
            "lr": 0.001,
            "n_epochs": n_epochs,
            "warmup_epochs": 30,
            "micro_lags": micro_lags,
        }
        
        S_t = torch.tensor(S_train, dtype=torch.float32, device=device)
        V_t = torch.tensor(V_train, dtype=torch.float32, device=device)
        lam_t = torch.tensor(lam_train, dtype=torch.float32, device=device)
        
        _ = train_model(model, S_t, V_t, lam_t, training_config,
                       representation=representation, T=T, K=K, vol_hat=vol_hat)
        
        # Collect latents
        mus0, zs0 = collect_latents(model, S0, V0, T, K, vol_hat, representation, device, micro_lags=micro_lags)
        mus1, zs1 = collect_latents(model, S1, V1, T, K, vol_hat, representation, device, micro_lags=micro_lags)
        
        # Train probes
        auc_z = train_regime_probe(zs0, zs1, window_size=1, seed=seed)
        auc_mu = train_regime_probe(mus0, mus1, window_size=1, seed=seed)
        
        print(f"{beta:<12.4f} {auc_z:<12.4f} {auc_mu:<12.4f}")
        
        results.append({
            "beta": float(beta),
            "probe_z_auc": float(auc_z),
            "probe_mu_auc": float(auc_mu),
            "representation": representation,
            "n_episodes": n_episodes,
            "horizon": horizon,
            "seed": seed,
            "leak_phi_r0": leak_phi_r0,
            "leak_phi_r1": leak_phi_r1,
        })
    
    print("-" * 60)
    
    # Check monotonicity
    auc_values = [r["probe_z_auc"] for r in results]
    is_monotonic = all(auc_values[i] >= auc_values[i+1] - 0.05 for i in range(len(auc_values)-1))
    
    print(f"\n📊 INTERPRETATION:")
    if is_monotonic:
        print(f"  ✓ Regime predictability decreases with β (monotonic trend)")
    else:
        print(f"  ⚠ Non-monotonic trend detected - check model training")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Regime Probe from Latent Z")
    parser.add_argument("--betas", type=float, nargs="+", default=[0.0, 0.01, 0.1, 1.0],
                        help="Beta values to evaluate")
    parser.add_argument("--n_episodes", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--representation", type=str, default="combined")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--leak_phi_r0", type=float, default=0.0)
    parser.add_argument("--leak_phi_r1", type=float, default=0.0)
    parser.add_argument("--micro_lags", type=int, default=0, help="Lagged micro signals for temporal inference")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    
    results = run_probe_sweep(
        betas=args.betas,
        n_episodes=args.n_episodes,
        horizon=args.horizon,
        representation=args.representation,
        seed=args.seed,
        n_epochs=args.n_epochs,
        leak_phi_r0=args.leak_phi_r0,
        leak_phi_r1=args.leak_phi_r1,
        micro_lags=args.micro_lags,
    )
    
    # Save results
    output_path = Path(args.output) if args.output else ROOT / "results" / "regime_probe_vs_beta.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_path}")


if __name__ == "__main__":
    main()
