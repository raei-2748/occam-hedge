
import torch
import numpy as np
from policies import FactorizedVariationalPolicy
from experiment_occam import bs_delta_call_torch, occam_features_torch

# We need occam_features_torch. It's in experiment_occam.py
# But I defined `occam_features_torch` inside experiment_occam without exporting?
# I should probably move features to `policies.py` or `utils.py` to share?
# Or just re-import from experiment_occam.

from experiment_occam import occam_features_torch

def evaluate_path_diagnostics(
    S: np.ndarray,
    V: np.ndarray,
    lam: np.ndarray,
    T: float,
    K: float,
    vol_hat: float,
    representation: str,
    weights_or_state_dict: dict,
) -> dict:
    """
    Evaluates policy and returns path-level diagnostics (Mean Volume, Total Turnover, Total Cost).
    Uses Torch model for VIB compatibility.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    S_t = torch.tensor(S, dtype=torch.float32, device=device)
    V_t = torch.tensor(V, dtype=torch.float32, device=device)
    lam_t = torch.tensor(lam, dtype=torch.float32, device=device)
    
    # Reconstruct Model
    input_dim = 2 if representation == "combined" else 1
    model = FactorizedVariationalPolicy(input_dim, latent_dim_per_feature=2).to(device)
    model.load_state_dict(weights_or_state_dict)
    model.eval()
    
    n_paths = S.shape[0]
    n_steps = V.shape[1] # Check logic again: V is (N, steps)
    tau_grid = torch.linspace(T, 0.0, n_steps + 1, device=device)
    
    a = torch.zeros(n_paths, device=device)
    path_turnover = torch.zeros(n_paths, device=device)
    path_cost = torch.zeros(n_paths, device=device)
    
    with torch.no_grad():
        for t in range(n_steps):
            feats = occam_features_torch(
                representation, 
                S_t[:, t], 
                torch.full((n_paths,), tau_grid[t], device=device), 
                V_t[:, t], 
                K, 
                vol_hat
            )
            
            # For diagnostics, we look at the mean action (deterministic) or sample?
            # Usually deterministic (mu) for analysis.
            # But the model returns (action, mus, logvars). 
            # The action computed by decoder from sampled z.
            action, _, _ = model(feats)
            
            a_new = torch.clamp(action, -5.0, 5.0)
            da = a_new - a
            
            path_turnover += torch.abs(da)
            path_cost += 0.5 * lam_t[:, t] * da**2
            
            a = a_new

    return {
        "turnover": path_turnover.cpu().numpy(),
        "cost": path_cost.cpu().numpy(),
        "volume": np.mean(V, axis=1)
    }
