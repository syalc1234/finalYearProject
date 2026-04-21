import math

import numpy as np
import pytest

from Bse_Explicit import bse_explicit_cn as cn


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


@pytest.mark.parametrize(
    ("solver", "option_type"),
    [
        (cn.bse_exp_call, "call"),
        (cn.bse_exp_put, "put"),
    ],
)
def test_explicit_solver_prices_match_black_scholes_at_spot(solver, option_type):
    sigma = 0.2
    rate = 0.05
    maturity = 1.0
    strike = 100.0
    spot = 100.0
    num_steps = 1000
    spatial_step = 2.0

    grid, t, s, elapsed = solver(sigma, rate, maturity, strike, num_steps, spatial_step, spot)

    s_max = max(3 * strike, 2.5 * spot)
    expected_prices = int(s_max / spatial_step)
    spot_index = int(round(spot / spatial_step))
    expected_price = _black_scholes_price(sigma, rate, maturity, strike, spot, option_type)

    assert grid.shape == (num_steps + 1, expected_prices + 1)
    assert np.allclose(t, np.linspace(0.0, maturity, num_steps + 1))
    assert np.allclose(s, np.linspace(0.0, s_max, expected_prices + 1))
    assert elapsed >= 0.0

    if option_type == "call":
        assert np.array_equal(grid[:, 0], np.zeros(num_steps + 1, dtype=np.float64))
        assert np.allclose(grid[-1, :], np.maximum(s - strike, 0.0))
        assert np.allclose(grid[:, -1], s_max - strike * np.exp(-rate * (maturity - t)))
    else:
        assert np.allclose(grid[:, 0], strike * np.exp(-rate * (maturity - t)))
        assert np.array_equal(grid[:, -1], np.zeros(num_steps + 1, dtype=np.float64))
        assert np.allclose(grid[-1, :], np.maximum(strike - s, 0.0))

    assert grid[0, spot_index] == pytest.approx(expected_price, abs=0.05)
