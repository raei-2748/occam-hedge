
from __future__ import annotations
import numpy as np
import math
import torch
import torch.nn as nn

def _norm_cdf(x):
    # vectorized math.erf, works everywhere
    x = np.asarray(x, dtype=float)
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def bs_delta_call(S: np.ndarray, K: float, tau: np.ndarray, vol: float, r: float = 0.0) -> np.ndarray:
    """
    Black–Scholes delta for a European call.
    S: (n_paths,) current price
    tau: (n_paths,) time-to-maturity in years
    """
    eps = 1e-12
    tau = np.maximum(tau, eps)
    vol = max(vol, 1e-12)

    d1 = (np.log((S + eps) / K) + (r + 0.5 * vol * vol) * tau) / (vol * np.sqrt(tau))
    return _norm_cdf(d1)


def policy_delta_hedge(S_t: np.ndarray, K: float, tau_t: np.ndarray, vol_hat: float) -> np.ndarray:
    """
    Payoff-anchored policy: hold delta shares.
    """
    return bs_delta_call(S_t, K=K, tau=tau_t, vol=vol_hat, r=0.0)


def policy_volume_reactive(
    S_t: np.ndarray,
    K: float,
    tau_t: np.ndarray,
    vol_hat: float,
    V_t: np.ndarray,
    k: float = 1.5,
) -> np.ndarray:
    """
    Microstructure-heavy policy: take delta, then scale aggressiveness by volume proxy.
    """
    base = bs_delta_call(S_t, K=K, tau=tau_t, vol=vol_hat, r=0.0)

    # Normalize volume within the cross-section at time t to a stable scale.
    V = V_t
    Vn = V / (np.median(V) + 1e-12)

    # Smooth, bounded scaling so it doesn't explode
    scale = 1.0 + k * np.tanh(np.log(Vn + 1e-12))
    return np.clip(base * scale, -5.0, 5.0)

# --- VIB Architecture ---

class VariationalEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x):
        h = self.net(x)
        return self.mu(h), self.logvar(h)

class FactorizedVariationalPolicy(nn.Module):
    """
    Implementation of Equation (4) from the paper: stochastic information bottleneck policy
    Factorized to allow independent monitoring of channel capacity (Greeks vs Micro).
    """
    def __init__(self, input_dim: int, latent_dim_per_feature: int = 1):
        super().__init__()
        self.input_dim = input_dim
        # Independent encoder for each feature to track SNR
        self.encoders = nn.ModuleList([
            VariationalEncoder(1, latent_dim_per_feature) for _ in range(input_dim)
        ])
        
        total_latent = input_dim * latent_dim_per_feature
        
        # Policy Network pi(z) -> action
        self.decoder = nn.Sequential(
            nn.Linear(total_latent, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.weights = None # Dummy for compatibility if needed, but we should remove dependency
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def forward(self, x):
        # x is (B, D)
        B, D = x.shape
        if D != self.input_dim:
            raise ValueError(f"Expected input dim {self.input_dim}, got {D}")
            
        mus, logvars = [], []
        zs = []
        
        for i in range(D):
            feat = x[:, i:i+1] # (B, 1)
            mu, logvar = self.encoders[i](feat)
            z = self.reparameterize(mu, logvar)
            
            mus.append(mu)
            logvars.append(logvar)
            zs.append(z)
            
        # Concat all latents
        z_cat = torch.cat(zs, dim=1)
        action = self.decoder(z_cat)
        
        return action.squeeze(-1), mus, logvars
