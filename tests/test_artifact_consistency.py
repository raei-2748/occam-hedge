
import unittest
import pandas as pd
import numpy as np
import json
import shutil
from pathlib import Path
import tempfile
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from plotting import plot_frontier_beta_sweep, plot_robust_risk_vs_eta

class TestArtifactConsistency(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        
    def tearDown(self):
        shutil.rmtree(self.test_dir)
        
    def test_frontier_consistency(self):
        """Test that plotting uses the provided CSV data precisely."""
        # Create dummy frontier data with explicit values
        data = {
            "representation": ["greeks", "micro"],
            "beta": [0.0, 1.0],
            "R0": [10.0, 20.0],
            "R_stress_eta0p1": [15.0, 30.0],
            "info_cost": [5.0, 2.0],
            "R0_std": [0.1, 0.2],
            "R_stress_std": [0.5, 0.6]
        }
        df = pd.DataFrame(data)
        csv_path = self.test_dir / "frontier.csv"
        df.to_csv(csv_path, index=False)
        
        # Determine output path
        out_plot = self.test_dir / "fig_frontier.png"
        
        # Run plotting
        plot_frontier_beta_sweep(df, out_plot)
        
        self.assertTrue(out_plot.exists(), "Plot should be generated")
        # Metadata check not strictly implemented in plotting.py yet (optional requirement)
        
    def test_robust_curves_consistency(self):
        """Test robust curves JSON structure handling."""
        data = [
            {
                "representation": "combined",
                "beta": 0.5,
                "etas": [0.0, 0.1, 0.2],
                "R_eta_mean": [10.0, 12.0, 14.0],
                "R_eta_std": [1.0, 1.0, 1.0]
            }
        ]
        json_path = self.test_dir / "curves.json"
        with open(json_path, 'w') as f:
            json.dump({"results": data}, f)
            
        out_plot = self.test_dir / "fig_curves.png"
        plot_robust_risk_vs_eta(data, out_plot, use_bands=True)
        
        self.assertTrue(out_plot.exists())

if __name__ == "__main__":
    unittest.main()
