import numpy as np


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
    from G_Brownian_Motion.Monte_Carlo import monte_carlo_option_price as _monte_carlo_option_price

    return _monte_carlo_option_price(
        S0,
        K,
        r,
        sigma,
        T,
        numofPaths,
        timeSteps=timeSteps,
        option_type=option_type,
        rng=rng,
    )


def monte_carlo_call_put(S0, K, r, sigma, T, numofPaths, timeSteps, rng=None):
    from G_Brownian_Motion.Monte_Carlo import monte_carlo_call_put as _monte_carlo_call_put

    return _monte_carlo_call_put(
        S0,
        K,
        r,
        sigma,
        T,
        numofPaths,
        timeSteps=timeSteps,
        rng=rng,
    )