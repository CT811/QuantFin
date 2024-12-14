import numpy as np

def MC_VAR(alpha, sigma, tau, s0):
    """Calculates a VaR value using the Monte Carlo simulation with Geometric Brownian Motion given a set of input parameters."""
    
    n_sims = 10000
    time_adjusted_sigma = sigma * np.sqrt(tau)
    simulated_returns = np.random.normal(loc=0, scale=time_adjusted_sigma, size=n_sims)

    simulated_prices = s0 * (1 + simulated_returns)
    losses = s0 - simulated_prices
    sorted_losses = np.sort(losses)

    var_index = int((1 - alpha) * n_sims)
    var = sorted_losses[var_index]

    return var