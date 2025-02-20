"""
Simulate a lognormal process and plot the results.
"""

import matplotlib.pyplot as plt
# import torch

import numpy as np

def simulate_stock_prices(S0, r, sigma, T, num_simulations):
    """
    Simulate stock prices at time T using the Black-Scholes model
    under a risk-neutral drift r (or a real-world drift mu).
    """
    # Generate standard normal variates
    Z = np.random.randn(num_simulations)
    
    # Compute simulated prices
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    return ST

def capped_gain_payoff(ST_values, S0, alpha, H):
    """
    Compute Pi(S_T) = alpha*(S_T - S0) if S_T < H,
                     alpha*(H - S0) otherwise.
    """
    payoff = np.where(ST_values < H,
                      alpha * (ST_values - S0),
                      alpha * (H - S0))
    return payoff

def estimate_expected_payoff(S0, alpha, H, r, sigma, T, num_simulations):
    """
    Estimate the expected payoff E[Pi(S_T)] by Monte Carlo simulation.
    """
    # Step 1: Simulate possible S_T values
    ST_values = simulate_stock_prices(S0, r, sigma, T, num_simulations)
    
    # Step 2: Compute payoff for each simulated S_T
    payoffs = capped_gain_payoff(ST_values, S0, alpha, H)
    
    # Step 3: Estimate the expected payoff
    return np.mean(payoffs)

import numpy as np
from scipy.integrate import quad

def lognormal_pdf(s, S0, r, sigma, T):
    """
    Returns the value of the lognormal pdf of S_T at s > 0,
    where ln(S_T) ~ Normal(m, sigma^2 T).
    """
    if s <= 0:
        return 0.0
    # Mean of ln(S_T)
    m = np.log(S0) + (r - 0.5 * sigma**2) * T
    # log(s) - mean
    z = (np.log(s) - m) / (sigma * np.sqrt(T))
    # pdf
    return (1.0 / (s * sigma * np.sqrt(2 * np.pi * T))) * np.exp(-0.5 * z**2)

def integrand_below_H(s, S0, alpha, H, r, sigma, T):
    """
    Integrand for s in [0, H], i.e. alpha * (s - S0) * f_{S_T}(s).
    """
    return alpha * (s - S0) * lognormal_pdf(s, S0, r, sigma, T)

def integrand_above_H(s, S0, alpha, H, r, sigma, T):
    """
    Integrand for s in [H, infty), i.e. alpha * (H - S0) * f_{S_T}(s).
    """
    return alpha * (H - S0) * lognormal_pdf(s, S0, r, sigma, T)

def expected_payoff_capped_gain(S0, alpha, H, r, sigma, T):
    """
    Computes the expected payoff of:
        Pi(S_T) = alpha*(S_T - S0) for S_T < H,
                  alpha*(H - S0)   for S_T >= H
    by direct numerical integration of the lognormal density.
    """
    # 1) Integrate from 0 to H: alpha*(s - S0)*f_{S_T}(s)
    integral_below, _ = quad(integrand_below_H, 
                             0, 
                             H, 
                             args=(S0, alpha, H, r, sigma, T))
    
    # 2) Integrate from H to infinity: alpha*(H - S0)*f_{S_T}(s)
    integral_above, _ = quad(integrand_above_H, 
                             H, 
                             np.inf, 
                             args=(S0, alpha, H, r, sigma, T))
    
    return integral_below + integral_above


def run_integrate():
    S0    = 100.0
    alpha = 0.8
    H     = 120.0
    r     = 0.05     # 'drift' or risk-free rate
    sigma = 0.2
    T     = 1.0      # time in years
    
    # Compute the expected payoff via direct integration
    result = expected_payoff_capped_gain(S0, alpha, H, r, sigma, T)
    print("Expected Payoff via Numerical Integration: {:.4f}".format(result))


def run_simulate():
    S0 = 100.0
    alpha = 0.8
    H = 120.0
    r = 0.05       # risk-free rate (or drift)
    sigma = 0.2    # volatility
    T = 1.0        # time (in years)
    num_sims = 10_000_000

    # Estimate the expected payoff
    estimated_payoff = estimate_expected_payoff(S0, alpha, H, r, sigma, T, num_sims)
    print("Estimated Expected Payoff:", estimated_payoff)
    return estimated_payoff

# Example usage
if __name__ == "__main__":

    run_integrate()
    run_simulate()

