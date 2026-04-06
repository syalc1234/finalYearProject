import time

import matplot2tikz
import numpy as np
from matplotlib import pyplot as plt

from Bse_Explicit.bse_analytical import bse_analytical


def geometricBrownianMotion(S0, mu, sigma, numofPaths, T, timeSteps, rng=None):
    dt = T / timeSteps
    generator = np.random.default_rng() if rng is None else rng

    # Brownian increments for each path and time step.
    dW = generator.normal(0.0, np.sqrt(dt), size=(numofPaths, timeSteps)).T

    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * dW
    increments = np.exp(drift + diffusion)

    St = np.vstack([np.ones(numofPaths), increments]).cumprod(axis=0) * S0
    return St


def european_option_payoff(ST, K, option_type="call"):
    option_type = option_type.lower()

    if option_type == "call":
        return np.maximum(ST - K, 0.0)
    if option_type == "put":
        return np.maximum(K - ST, 0.0)

    raise ValueError("option_type must be 'call' or 'put'")


def simulate_terminal_prices(S0, r, sigma, T, numofPaths, rng=None):
    generator = np.random.default_rng() if rng is None else rng
    z = generator.normal(0.0, 1.0, size=numofPaths)
    drift = (r - 0.5 * sigma**2) * T
    diffusion = sigma * np.sqrt(T) * z
    return S0 * np.exp(drift + diffusion)


def option_price_from_terminal_prices(ST, K, r, T, option_type="call"):
    payoff = european_option_payoff(ST, K, option_type=option_type)
    discount_factor = np.exp(-r * T)
    discounted_payoff = discount_factor * payoff

    return {
        "price": discounted_payoff.mean(),
        "std_error": discounted_payoff.std(ddof=1) / np.sqrt(ST.size),
    }


def monte_carlo_option_price(S0, K, r, sigma, T, numofPaths, timeSteps, option_type="call", rng=None):
    # For European vanilla options under GBM, sampling the terminal distribution
    # directly avoids allocating the full path matrix.
    ST = simulate_terminal_prices(S0, r, sigma, T, numofPaths, rng=rng)
    result = option_price_from_terminal_prices(ST, K, r, T, option_type=option_type)

    return result


def monte_carlo_call_put(S0, K, r, sigma, T, numofPaths, timeSteps, rng=None):
    generator = np.random.default_rng() if rng is None else rng
    ST = simulate_terminal_prices(S0, r, sigma, T, numofPaths, rng=generator)

    call = option_price_from_terminal_prices(ST, K, r, T, option_type="call")
    put = option_price_from_terminal_prices(ST, K, r, T, option_type="put")

    return {"call": call, "put": put}


def main123():

    S0 = 274.80
    K = 275.00
    r = 0.0375
    sigma = 0.1730
    T = 1
    rng = np.random.default_rng(42)
    time_taken = np.zeros(7)
    gridSize = [ 65536, 262144, 1048576, 4194304, 16777216, 33554432]

    times = []
    prices = []

    for dS in gridSize:
        start = time.time()
        results = monte_carlo_call_put(
            S0=S0,
            K=K,
            r=r,
            sigma=sigma,
            T=T,
            numofPaths=dS,
            timeSteps=128,
            rng=np.random.default_rng(42),
        )
        end = time.time() - start
        prices.append(float(results['call']['price']))
        times.append(float(end))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(gridSize, times, marker="o", label="Python")
    axes[0].legend()
    axes[0].set_xlabel("Number of Monte Carlo paths")
    axes[0].set_ylabel("Time taken (seconds)")
    axes[0].set_title("Runtime vs number of Monte Carlo paths")
    axes[0].grid(True)

    fig.show()

    prices = np.array(prices)
    print(prices)
    print(np.multiply(times, 1000))
    bs = float(bse_analytical(sigma, r, T, K, S0))
    print(bs)

def main():
    CPU = [23.96317248, 23.98589092, 23.98139959, 23.9940673, 23.96806923, 23.97552303]
    GPU = [23.9657,23.9837, 23.967, 23.9774, 23.9756, 23.978]
    bs = 23.9785783872683
    gridSize = [65536, 262144, 1048576, 4194304, 16777216, 33554432]
    times_cpu = [3.03077698, 11.97242737, 50.22287369, 196.63167, 836.00306511, 2006.6242218 ]
    times_gpu = [9.07162,9.67373, 9.3143, 9.69728, 10.4386, 11.0386]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(gridSize, times_cpu, marker="o", label="Python")
    axes[0].plot(gridSize, times_gpu, marker="o", label="GPU")
    axes[0].legend()
    axes[0].set_xlabel("Number of Monte Carlo paths base 2")
    axes[0].set_ylabel("Time taken (ms)")
    axes[0].set_title("Runtime vs number of Monte Carlo paths")
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].grid(True)

    axes[1].plot()

    axes[1].plot(gridSize, CPU, marker="o", linewidth=2, label="CPU (Python) Price")
    axes[1].plot(gridSize, GPU, marker="o", linewidth=2, label="GPU Price")
    axes[1].axhline(bs, linestyle="--", linewidth=2, label="Analytical (Black-Scholes)")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel(r"Spatial step size $\Delta S$")
    axes[1].set_ylabel("Option price V(S, t=0)")
    axes[1].set_title("CPU and GPU Monte Carlo price estimates vs number of paths")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    matplot2tikz.save("cpuVsGPUMCarlo.tex", encoding="utf-8")

    fig.show()

if __name__ == "__main__":
    main()


"""
CPU:
[23.96317248 23.98589092 23.98139959 23.9940673  23.96806923 23.97552303]

GPU:
[23.9657,23.9837, 23.967, 23.9774, 23.9756, 23.978]

23.9785783872683

"""