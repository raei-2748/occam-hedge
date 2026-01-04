# Project Context: Occam's Hedge

## Core Research Question
How do we mitigate "wrong-way trading" caused by **Semantic Inversion** in market microstructure proxies under regime shifts?

## The Mechanism
- **Semantic Inversion**: An equilibrium shift where a proxy (e.g., volume) that signaled "cheap trading" in Regime 0 signals "expensive trading" in Regime 1.
- **Information Failure**: Safe use of sign-flipping signals requires 1 bit of latent regime information ($I(R;Z) \ge \log 2 - h(p_e)$).
- **The Solution**: Occam's Hedge uses a **Variational Information Bottleneck (VIB)** to price this information. Tightening the budget ($\beta$) forces the agent to rely on payoff-anchored structural invariants (Greeks) rather than fragile equilibrium proxies.

## Experimental Pipeline
1. **`src/world.py`**: Simulator implementing Heston dynamics and the sign-flipping impact elasticity $\lambda_{r,t}(\mathrm{Vol}_t)$.
2. **`src/policies.py`**: Implementation of the Factorized Variational Policy.
3. **`src/risk.py`**: Robust risk evaluation using the Donsker-Varadhan dual form.
4. **Oracle Disambiguation**: A "smoking gun" test where the agent is given the true regime label to prove the failure is informational.