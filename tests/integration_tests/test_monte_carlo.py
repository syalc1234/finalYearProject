import math

import numpy as np
import pytest

from Brownian_Motion import Geometric_BM as gbm


def _normal_cdf(value):
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _black_scholes_price(sigma, rate, maturity, strike, spot, option_type):
    d1 = (
        math.log(spot / strike) + (rate + 0.5 * sigma**2) * maturity
    ) / (sigma * math.sqrt(maturity))
    d2 = d1 - sigma * math.sqrt(maturity)

    if option_type == "call":
        return spot * _normal_cdf(d1) - strike * math.exp(-rate * maturity) * _normal_cdf(d2)

    return strike * math.exp(-rate * maturity) * _normal_cdf(-d2) - spot * _normal_cdf(-d1)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_seeded_monte_carlo_price_converges_to_black_scholes(option_type):
    spot = 100.0
    strike = 105.0
    rate = 0.04
    sigma = 0.25
    maturity = 1.25
    num_paths = 200_000
    rng = np.random.default_rng(20260421)

    result = gbm.monte_carlo_option_price(
        spot,
        strike,
        rate,
        sigma,
        maturity,
        numofPaths=num_paths,
        timeSteps=365,
        option_type=option_type,
        rng=rng,
    )

    expected_price = _black_scholes_price(sigma, rate, maturity, strike, spot, option_type)
    tolerance = max(0.10, 5.0 * result["std_error"])

    assert result["price"] == pytest.approx(expected_price, abs=tolerance)
    assert result["std_error"] > 0.0
