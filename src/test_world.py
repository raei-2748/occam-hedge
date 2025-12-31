import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from world import simulate_heston_signflip

# Save figures relative to the repository root, not the current working directory.
ROOT = Path(__file__).resolve().parents[1]
FIGDIR = ROOT / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

def corr(x, y):
    return np.corrcoef(x.reshape(-1), y.reshape(-1))[0, 1]

S0, v0, V0, lam0, _ = simulate_heston_signflip(regime=0, n_paths=3000, n_steps=100, seed=0)
S1, v1, V1, lam1, _ = simulate_heston_signflip(regime=1, n_paths=3000, n_steps=100, seed=0)

print("corr(V, lambda) regime0 =", corr(V0, lam0))
print("corr(V, lambda) regime1 =", corr(V1, lam1))

# scatter (downsample)
idx0 = np.random.choice(V0.size, size=5000, replace=False)
idx1 = np.random.choice(V1.size, size=5000, replace=False)

plt.figure()
plt.scatter(V0.reshape(-1)[idx0], lam0.reshape(-1)[idx0], s=6)
plt.xlabel("Volume proxy")
plt.ylabel("Impact coefficient (lambda)")
plt.title("Regime 0: high volume => low impact")
out0 = FIGDIR / "signflip_heston_regime0.png"
plt.savefig(out0, dpi=200)
print(f"Saved {out0}")

plt.figure()
plt.scatter(V1.reshape(-1)[idx1], lam1.reshape(-1)[idx1], s=6)
plt.xlabel("Volume proxy")
plt.ylabel("Impact coefficient (lambda)")
plt.title("Regime 1: high volume => high impact")
out1 = FIGDIR / "signflip_heston_regime1.png"
plt.savefig(out1, dpi=200)
print(f"Saved {out1}")
