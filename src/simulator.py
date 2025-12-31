import numpy as np


def simulate_signflip(regime: int, n: int = 50_000, seed: int = 0):
    """
    Returns (volume, impact_lambda) samples.

    regime=0: high volume => low impact  (lambda ~ 1/vol)
    regime=1: high volume => high impact (lambda ~ vol)
    """
    rng = np.random.default_rng(seed + 1000 * regime)

    # positive volume proxy (lognormal is a nice simple choice)
    vol = rng.lognormal(mean=0.0, sigma=0.6, size=n)

    # add a latent "stress/tightness" multiplier (optional but realistic)
    latent = rng.lognormal(mean=0.0, sigma=0.4 if regime == 0 else 0.8, size=n)

    if regime == 0:
        lam = latent / (vol + 1e-6)
    else:
        lam = latent * vol

    return vol, lam