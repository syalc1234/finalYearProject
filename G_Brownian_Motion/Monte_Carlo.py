import numpy as np

from G_Brownian_Motion.Geometric_BM import (
    option_price_from_terminal_prices,
    simulate_terminal_prices,
)


class MonteCarlo:
    # Monte Carlo pricer for European options under a GBM stock model.
    def __init__(
        self,
        spot,
        strike,
        rate,
        sigma,
        maturity,
        num_paths,
        time_steps=365,
        rng=None,
    ):
        self.spot = spot
        self.strike = strike
        self.rate = rate
        self.sigma = sigma
        self.maturity = maturity
        self.num_paths = num_paths
        self.time_steps = time_steps
        self.rng = np.random.default_rng() if rng is None else rng

    def terminal_prices(self):
        # European vanilla options only need the asset value at maturity.
        return simulate_terminal_prices(
            self.spot,
            self.rate,
            self.sigma,
            self.maturity,
            self.num_paths,
            rng=self.rng,
        )

    def price(self, option_type="call"):
        # Price one option type by discounting the simulated terminal payoff.
        terminal_prices = self.terminal_prices()
        return option_price_from_terminal_prices(
            terminal_prices,
            self.strike,
            self.rate,
            self.maturity,
            option_type=option_type,
        )

    def call_price(self):
        return self.price(option_type="call")

    def put_price(self):
        return self.price(option_type="put")

    def call_put_prices(self):
        # Use one set of terminal prices so the call and put estimates are comparable.
        terminal_prices = self.terminal_prices()
        call = option_price_from_terminal_prices(
            terminal_prices,
            self.strike,
            self.rate,
            self.maturity,
            option_type="call",
        )
        put = option_price_from_terminal_prices(
            terminal_prices,
            self.strike,
            self.rate,
            self.maturity,
            option_type="put",
        )

        return {"call": call, "put": put}


def monte_carlo_option_price(
    S0,
    K,
    r,
    sigma,
    T,
    numofPaths,
    timeSteps=365,
    option_type="call",
    rng=None,
):
    # Backwards-compatible function wrapper around the class API.
    pricer = MonteCarlo(S0, K, r, sigma, T, numofPaths, time_steps=timeSteps, rng=rng)
    return pricer.price(option_type=option_type)


def monte_carlo_call_put(S0, K, r, sigma, T, numofPaths, timeSteps=365, rng=None):
    # Backwards-compatible wrapper for pricing both vanilla option types.
    pricer = MonteCarlo(S0, K, r, sigma, T, numofPaths, time_steps=timeSteps, rng=rng)
    return pricer.call_put_prices()


Monte_Carlo = MonteCarlo
