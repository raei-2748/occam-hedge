import sys
from pathlib import Path

# add src/ to Python path
sys.path.append(str(Path(__file__).parent))

import numpy as np
import matplotlib.pyplot as plt
from risk import robust_risk_dv

# Fake baseline losses: mostly small, sometimes big (heavy tail)
rng = np.random.default_rng(0)
losses = rng.normal(loc=0.0, scale=1.0, size=50_000)
# simpler heavy tail injection:
mask = rng.random(losses.shape) < 0.02
losses[mask] += rng.normal(loc=8.0, scale=2.0, size=mask.sum())

etas = np.array([0.0, 0.01, 0.02, 0.05, 0.1, 0.2])

R = [robust_risk_dv(losses, eta=e) for e in etas]

# Save plot to file (reliable on macOS)
plt.plot(etas, R, marker="o")
plt.xlabel("eta (KL stress budget)")
plt.ylabel("R_eta (robust risk)")
plt.title("KL-stress robust risk increases with eta")

from pathlib import Path

out_path = Path(__file__).resolve().parents[1] / "figures" / "robust_risk_curve.png"
print("About to save to:", out_path)

plt.savefig(out_path, dpi=200)
print("Saved OK")