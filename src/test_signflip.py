import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from simulator import simulate_signflip

# Save figures relative to the repository root, not the current working directory.
ROOT = Path(__file__).resolve().parents[1]
FIGDIR = ROOT / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

vol0, lam0 = simulate_signflip(regime=0, n=30_000, seed=0)
vol1, lam1 = simulate_signflip(regime=1, n=30_000, seed=0)

# quick correlation check (should flip sign)
corr0 = np.corrcoef(vol0, lam0)[0, 1]
corr1 = np.corrcoef(vol1, lam1)[0, 1]
print("corr(volume, impact) regime 0 =", corr0)
print("corr(volume, impact) regime 1 =", corr1)

# plot (downsample for speed/clarity)
idx0 = np.random.choice(len(vol0), size=3000, replace=False)
idx1 = np.random.choice(len(vol1), size=3000, replace=False)

plt.figure()
plt.scatter(vol0[idx0], lam0[idx0], s=6)
plt.xlabel("Volume proxy")
plt.ylabel("Impact coefficient (lambda)")
plt.title("Regime 0: high volume => low impact")

out0 = FIGDIR / "signflip_regime0.png"
plt.savefig(out0, dpi=200)
print(f"Saved {out0}")

plt.figure()
plt.scatter(vol1[idx1], lam1[idx1], s=6)
plt.xlabel("Volume proxy")
plt.ylabel("Impact coefficient (lambda)")
plt.title("Regime 1: high volume => high impact")

out1 = FIGDIR / "signflip_regime1.png"
plt.savefig(out1, dpi=200)
print(f"Saved {out1}")
