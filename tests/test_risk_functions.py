from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from risk import robust_expectation, expected_shortfall, es_loss, optimize_q_es


def test_robust_expectation_properties():
    rng = np.random.default_rng(0)
    losses = rng.normal(loc=0.0, scale=1.0, size=4000)
    tail = rng.random(losses.shape) < 0.02
    losses[tail] += rng.normal(loc=6.0, scale=1.0, size=tail.sum())

    eta_low = 0.05
    eta_mid = 0.125
    eta_high = 0.2

    lambdas = np.logspace(-4, 2, 400)
    r0 = robust_expectation(losses, eta=0.0)
    r_low = robust_expectation(losses, eta=eta_low, lambdas=lambdas)
    r_mid = robust_expectation(losses, eta=eta_mid, lambdas=lambdas)
    r_high = robust_expectation(losses, eta=eta_high, lambdas=lambdas)

    assert r_low >= r0 - 1e-8
    assert r_mid >= r_low - 1e-8
    assert r_high >= r_mid - 1e-8

    mean_loss = float(np.mean(losses))
    assert r_low >= mean_loss - 1e-6

    convex_bound = 0.5 * (r_low + r_high)
    assert r_mid <= convex_bound + 5e-2


def test_expected_shortfall_matches_grid():
    rng = np.random.default_rng(1)
    losses = rng.standard_t(df=5, size=5000)
    gamma = 0.9

    es_fast = expected_shortfall(losses, gamma)

    q_grid = np.quantile(losses, np.linspace(0.5, 0.995, 200))
    es_grid = min(float(np.mean(es_loss(losses, q, gamma))) for q in q_grid)

    assert abs(es_fast - es_grid) <= 1e-2


def test_optimize_q_es_grid_consistency():
    rng = np.random.default_rng(2)
    losses = rng.normal(size=3000)
    gamma = 0.95

    q_grid = np.quantile(losses, np.linspace(0.7, 0.99, 100))
    q_star = optimize_q_es(losses, gamma, q_grid=q_grid)

    es_star = float(np.mean(es_loss(losses, q_star, gamma)))
    es_grid = min(float(np.mean(es_loss(losses, q, gamma))) for q in q_grid)

    assert es_star <= es_grid + 1e-10
