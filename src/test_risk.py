
import unittest
import numpy as np
import math
from risk import robust_expectation, robust_risk_dv

class TestRisk(unittest.TestCase):

    def test_gaussian_analytic(self):
        # Analytic result for Gaussian: R_eta(N(mu,1)) = mu + sqrt(2*eta)
        mu = 0.0
        for eta in [0.0, 0.1, 0.5, 1.0]:
            if eta == 0.0:
                expected = mu
            else:
                expected = mu + math.sqrt(2 * eta)
            
            rng = np.random.RandomState(42)
            losses = mu + rng.randn(100_000)
            
            # We need a fine grid for accuracy
            lambdas = np.logspace(-2, 1, 500)
            
            val = robust_risk_dv(losses, eta=eta, lambdas=lambdas)
            
            # Tolerance: Monte Carlo error + Optimization grid error
            err = abs(val - expected)
            print(f"Eta={eta}: Expected={expected:.4f}, Got={val:.4f}, Err={err:.4f}")
            self.assertTrue(err < 0.05, f"Eta={eta}: Expected {expected}, got {val}")

    def test_gaussian_shifted(self):
        # R_eta(N(mu, 1)) = mu + sqrt(2*eta)
        mu = 2.0
        eta = 0.5
        expected = mu + math.sqrt(2 * eta)
        
        rng = np.random.RandomState(42)
        losses = mu + rng.randn(100_000)
        lambdas = np.logspace(-2, 1, 500)
        
        val = robust_risk_dv(losses, eta=eta, lambdas=lambdas)
        self.assertTrue(abs(val - expected) < 0.05)

    def test_lognormal(self):
        # Lognormal(0, 0.5).
        rng = np.random.RandomState(42)
        s = 0.5
        losses = rng.lognormal(mean=0.0, sigma=s, size=100_000)
        
        val_0 = robust_risk_dv(losses, eta=0.0)
        mean_ref = np.mean(losses)
        
        self.assertTrue(abs(val_0 - mean_ref) < 1e-6)
        
        val_pos = robust_risk_dv(losses, eta=0.1)
        self.assertTrue(val_pos > val_0, "Robust risk should be >= risk")

    def test_robust_expectation_raises(self):
        with self.assertRaises(ValueError):
            robust_expectation([], eta=0.1)
        with self.assertRaises(ValueError):
            robust_expectation([1, 2], eta=-0.1)
        with self.assertRaises(ValueError):
            robust_expectation([1, 2], eta=0.1, lambdas=[-1])

if __name__ == '__main__':
    unittest.main()
