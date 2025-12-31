import numpy as np


def logsumexp(x: np.ndarray) -> float:
    """
    Stable computation of log(sum(exp(x))).
    """
    x = np.asarray(x, dtype=float)
    m = np.max(x)
    if not np.isfinite(m):
        return float(m)
    return float(m + np.log(np.sum(np.exp(x - m))))


def log_mean_exp(x: np.ndarray) -> float:
    """
    Stable computation of log(mean(exp(x))).
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        raise ValueError("log_mean_exp requires at least one sample")
    return logsumexp(x) - np.log(x.size)


def robust_expectation(loss_samples: np.ndarray, eta: float, lambdas: np.ndarray | None = None) -> float:
    """
    Compute KL-stress robust expectation via the DV dual:
        R_eta = inf_{lambda>0} (1/lambda) * ( log E[exp(lambda * loss)] + eta )
    """
    losses = np.asarray(loss_samples, dtype=float)
    if losses.size == 0:
        raise ValueError("robust_expectation requires at least one sample")
    if eta < 0.0:
        raise ValueError("eta must be nonnegative")
    if eta == 0.0:
        return float(np.mean(losses))

    if lambdas is None:
        lambdas = np.logspace(-4, 2, 200)
    lambdas = np.asarray(lambdas, dtype=float)
    lambdas = lambdas[lambdas > 0.0]
    if lambdas.size == 0:
        raise ValueError("lambdas must contain positive values")

    vals = []
    for lam in lambdas:
        lme = log_mean_exp(lam * losses)
        vals.append((lme + eta) / lam)

    return float(np.min(vals))


def robust_risk_dv(losses: np.ndarray, eta: float, lambdas: np.ndarray | None = None) -> float:
    """
    Backwards-compatible wrapper for KL-stress robust expectation.
    """
    return robust_expectation(losses, eta=eta, lambdas=lambdas)


def es_loss(losses: np.ndarray, q: float, gamma: float) -> np.ndarray:
    losses = np.asarray(losses, dtype=float)
    if not 0.0 < gamma < 1.0:
        raise ValueError("gamma must be in (0, 1)")
    return q + (1.0 / (1.0 - gamma)) * np.maximum(losses - q, 0.0)


def optimize_q_es(losses: np.ndarray, gamma: float, q_grid: np.ndarray | None = None) -> float:
    """
    Optimize q in the convex ES representation.
    """
    losses = np.asarray(losses, dtype=float)
    if losses.size == 0:
        raise ValueError("optimize_q_es requires at least one sample")
    if not 0.0 < gamma < 1.0:
        raise ValueError("gamma must be in (0, 1)")
    if q_grid is None:
        # Default behavior: sample quantile
        return float(np.quantile(losses, gamma))
    
    q_grid = np.asarray(q_grid, dtype=float)
    if q_grid.size == 0:
        raise ValueError("q_grid must contain values")
    vals = [float(np.mean(es_loss(losses, q, gamma))) for q in q_grid]
    return float(q_grid[int(np.argmin(vals))])


def expected_shortfall(losses: np.ndarray, gamma: float, q: float | None = None) -> float:
    """
    Expected Shortfall using the convex representation with optimized q.
    """
    losses = np.asarray(losses, dtype=float)
    q_star = optimize_q_es(losses, gamma) if q is None else float(q)
    return float(np.mean(es_loss(losses, q_star, gamma)))


def _default_q_grid(losses: np.ndarray, gamma: float, n_grid: int = 41) -> np.ndarray:
    lo = max(0.5, gamma - 0.2)
    hi = min(0.995, gamma + 0.05)
    grid = np.linspace(lo, hi, n_grid)
    return np.unique(np.quantile(losses, grid))


def robust_es_kl(
    losses: np.ndarray,
    eta: float,
    gamma: float,
    lambdas: np.ndarray | None = None,
    q_grid: np.ndarray | None = None,
) -> float:
    """
    R_eta = min_q sup_Q E_Q[ell_q(Y)] with ell_q from ES.
    """
    losses = np.asarray(losses, dtype=float)
    if q_grid is None:
        q_grid = _default_q_grid(losses, gamma)
    vals = []
    for q in q_grid:
        lq = es_loss(losses, float(q), gamma)
        vals.append(robust_expectation(lq, eta=eta, lambdas=lambdas))
    return float(np.min(vals))


# --- Torch Support ---
try:
    import torch
    
    def robust_risk_torch(
        losses: torch.Tensor, 
        eta: float, 
        lambdas: torch.Tensor
    ) -> torch.Tensor:
        """
        Differentiable Robust Risk (KL-stress) using Torch.
        losses: (B,)
        lambdas: (N_lam,)
        Returns scalar tensor.
        """
        # (N_lam, B)
        scaled = lambdas.unsqueeze(1) * losses.unsqueeze(0)
        
        # log_mean_exp over batch (dim 1)
        # LME = log(mean(exp(X))) = log(sum(exp(X))) - log(N)
        lse = torch.logsumexp(scaled, dim=1) # (N_lam,)
        lme = lse - np.log(losses.shape[0])
        
        # Dual function: g(lam) = (lme + eta) / lam
        g = (lme + eta) / lambdas
        
        # Return min over lambdas
        return torch.min(g)

    def es_loss_torch(losses: torch.Tensor, q: torch.Tensor, gamma: float) -> torch.Tensor:
        """
        Sample-wise ES loss: q + (1/(1-gamma)) * max(0, L - q)
        Returns vector of same shape as losses.
        """
        return q + (1.0 / (1.0 - gamma)) * torch.nn.functional.relu(losses - q)

except ImportError:
    pass
