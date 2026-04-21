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


def gbm_lognormal_pdf(x, S0, mu, sigma, T):
    x = np.asarray(x, dtype=float)
    pdf = np.zeros_like(x)

    positive_mask = x > 0.0
    if not np.any(positive_mask):
        return pdf

    variance = sigma**2 * T
    log_mean = np.log(S0) + (mu - 0.5 * sigma**2) * T
    denominator = x[positive_mask] * np.sqrt(2.0 * np.pi * variance)
    exponent = -((np.log(x[positive_mask]) - log_mean) ** 2) / (2.0 * variance)
    pdf[positive_mask] = np.exp(exponent) / denominator
    return pdf


def normal_pdf(x, mean, std):
    x = np.asarray(x, dtype=float)
    return np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2.0 * np.pi))



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
    from Brownian_Motion.plots import plot_gbm_lognormal_distribution

    S0 = 274.80
    sigma = 0.1730
    T = 1
    fig, ax, ST = plot_gbm_lognormal_distribution(S0, 0.0375, sigma, T, numofPaths=20000000, bins=75, rng=None, ax=None, tikz_filename="../plots_tex/logNormaldist.tex")
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
