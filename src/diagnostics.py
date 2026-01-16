
import torch
import numpy as np
from policies import FactorizedVariationalPolicy
from experiment_occam import bs_delta_call_torch, occam_features_torch

# We need occam_features_torch. It's in experiment_occam.py
# But I defined `occam_features_torch` inside experiment_occam without exporting?
# I should probably move features to `policies.py` or `utils.py` to share?
# Or just re-import from experiment_occam.

from experiment_occam import occam_features_torch
from features import get_feature_dim

def evaluate_path_diagnostics(
    S: np.ndarray,
    V: np.ndarray,
    lam: np.ndarray,
    T: float,
    K: float,
    vol_hat: float,
    representation: str,
    weights_or_state_dict: dict,
    micro_lags: int = 0,
    include_prev_action: bool = False
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
    input_dim = get_feature_dim(representation, include_prev_action=include_prev_action, micro_lags=micro_lags)
    model = FactorizedVariationalPolicy(input_dim, latent_dim_per_feature=8).to(device)
    model.load_state_dict(weights_or_state_dict)
    model.eval()
    
    n_paths = S.shape[0]
    n_steps = V.shape[1] # Check logic again: V is (N, steps)
    tau_grid = torch.linspace(T, 0.0, n_steps + 1, device=device)
    
    a = torch.zeros(n_paths, device=device)
    path_turnover = torch.zeros(n_paths, device=device)
    path_cost = torch.zeros(n_paths, device=device)
    
    # Initialize micro buffer for lags
    micro_buffer = []

    with torch.no_grad():
        for t in range(n_steps):
            # Prepare V_history
            if micro_lags > 0 and representation in ["micro", "combined"]:
                if len(micro_buffer) >= micro_lags:
                    V_history = torch.stack(micro_buffer[-micro_lags:][::-1], dim=1)
                elif len(micro_buffer) > 0:
                     available = torch.stack(micro_buffer[::-1], dim=1)
                     padding = torch.zeros(n_paths, micro_lags - len(micro_buffer), device=device)
                     V_history = torch.cat([available, padding], dim=1)
                else:
                    V_history = None
            else:
                 V_history = None
            
            # Prepare prev action
            a_prev = a if include_prev_action else None

            feats = occam_features_torch(
                representation, 
                S_t[:, t], 
                torch.full((n_paths,), tau_grid[t], device=device), 
                V_t[:, t], 
                K, 
                vol_hat,
                micro_lags=micro_lags,
                include_prev_action=include_prev_action,
                V_history=V_history,
                a_prev=a_prev
            )
            
            # Update Buffer
            if micro_lags > 0 and representation in ["micro", "combined"]:
                from features import micro_signal_torch
                current_micro = micro_signal_torch(V_t[:, t])
                micro_buffer.append(current_micro)
                if len(micro_buffer) > micro_lags:
                    micro_buffer.pop(0)

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
