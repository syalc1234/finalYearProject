import math

import numpy as np
import pytest

from G_Brownian_Motion import Geometric_BM as gbm


class FixedRng:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)
        self.calls = []

    def normal(self, loc, scale, size):
        self.calls.append((loc, scale, size))
        return self.values.reshape(size)


def test_geometric_brownian_motion_uses_expected_path_formula():
    s0 = 100.0
    mu = 0.08
    sigma = 0.2
    maturity = 1.0
    time_steps = 3
    num_paths = 2
    dt = maturity / time_steps
    raw_increments = np.array(
        [
            [0.1, -0.2, 0.0],
            [0.3, 0.2, -0.1],
        ]
    )
    rng = FixedRng(raw_increments)

    paths = gbm.geometricBrownianMotion(
        s0,
        mu,
        sigma,
        num_paths,
        maturity,
        time_steps,
        rng=rng,
    )

    expected_step_returns = np.exp((mu - 0.5 * sigma**2) * dt + sigma * raw_increments.T)
    expected_paths = np.vstack([np.ones(num_paths), expected_step_returns]).cumprod(axis=0) * s0

    assert paths.shape == (time_steps + 1, num_paths)
    assert np.array_equal(paths[0], np.full(num_paths, s0))
    assert np.allclose(paths, expected_paths)
    assert rng.calls == [(0.0, math.sqrt(dt), (num_paths, time_steps))]


def test_geometric_brownian_motion_zero_volatility_is_deterministic():
    s0 = 50.0
    mu = 0.1
    maturity = 2.0
    time_steps = 4
    num_paths = 3
    dt = maturity / time_steps
    rng = FixedRng(np.zeros((num_paths, time_steps)))

    paths = gbm.geometricBrownianMotion(
        s0,
        mu,
        sigma=0.0,
        numofPaths=num_paths,
        T=maturity,
        timeSteps=time_steps,
        rng=rng,
    )

    expected_times = np.arange(time_steps + 1) * dt
    expected = s0 * np.exp(mu * expected_times)

    assert np.allclose(paths, np.repeat(expected[:, None], num_paths, axis=1))


@pytest.mark.parametrize(
    ("option_type", "expected"),
    [
        ("call", np.array([0.0, 0.0, 5.0])),
        ("CALL", np.array([0.0, 0.0, 5.0])),
        ("put", np.array([10.0, 0.0, 0.0])),
        ("PUT", np.array([10.0, 0.0, 0.0])),
    ],
)
def test_european_option_payoff(option_type, expected):
    terminal_prices = np.array([90.0, 100.0, 105.0])

    payoff = gbm.european_option_payoff(terminal_prices, K=100.0, option_type=option_type)

    assert np.array_equal(payoff, expected)


def test_european_option_payoff_rejects_unknown_option_type():
    with pytest.raises(ValueError, match="option_type must be 'call' or 'put'"):
        gbm.european_option_payoff(np.array([100.0]), K=100.0, option_type="straddle")


def test_simulate_terminal_prices_uses_closed_form_gbm_distribution():
    s0 = 100.0
    rate = 0.05
    sigma = 0.2
    maturity = 1.5
    z_values = np.array([-1.0, 0.0, 1.0])
    rng = FixedRng(z_values)

    terminal_prices = gbm.simulate_terminal_prices(
        s0,
        rate,
        sigma,
        maturity,
        numofPaths=z_values.size,
        rng=rng,
    )

    expected = s0 * np.exp((rate - 0.5 * sigma**2) * maturity + sigma * math.sqrt(maturity) * z_values)

    assert np.allclose(terminal_prices, expected)
    assert rng.calls == [(0.0, 1.0, z_values.size)]


def test_gbm_lognormal_pdf_matches_formula_and_zeros_nonpositive_values():
    x = np.array([-1.0, 0.0, 80.0, 100.0, 120.0])
    s0 = 100.0
    mu = 0.05
    sigma = 0.2
    maturity = 1.0

    pdf = gbm.gbm_lognormal_pdf(x, s0, mu, sigma, maturity)

    positive_x = x[x > 0.0]
    variance = sigma**2 * maturity
    log_mean = math.log(s0) + (mu - 0.5 * sigma**2) * maturity
    expected_positive_pdf = np.exp(-((np.log(positive_x) - log_mean) ** 2) / (2.0 * variance)) / (
        positive_x * np.sqrt(2.0 * np.pi * variance)
    )

    assert np.array_equal(pdf[:2], np.zeros(2))
    assert np.allclose(pdf[2:], expected_positive_pdf)


def test_gbm_lognormal_pdf_returns_zeros_when_all_values_are_nonpositive():
    x = np.array([-5.0, 0.0])

    pdf = gbm.gbm_lognormal_pdf(x, S0=100.0, mu=0.05, sigma=0.2, T=1.0)

    assert np.array_equal(pdf, np.zeros_like(x))


def test_normal_pdf_matches_standard_formula():
    x = np.array([-1.0, 0.0, 1.0])

    pdf = gbm.normal_pdf(x, mean=0.0, std=1.0)

    expected = np.exp(-0.5 * x**2) / np.sqrt(2.0 * np.pi)
    assert np.allclose(pdf, expected)


@pytest.mark.parametrize(
    ("option_type", "expected_payoff"),
    [
        ("call", np.array([0.0, 0.0, 20.0])),
        ("put", np.array([20.0, 0.0, 0.0])),
    ],
)
def test_option_price_from_terminal_prices_discounts_mean_payoff_and_std_error(option_type, expected_payoff):
    terminal_prices = np.array([80.0, 100.0, 120.0])
    strike = 100.0
    rate = 0.05
    maturity = 2.0
    discount_factor = math.exp(-rate * maturity)

    result = gbm.option_price_from_terminal_prices(
        terminal_prices,
        strike,
        rate,
        maturity,
        option_type=option_type,
    )

    discounted_payoff = discount_factor * expected_payoff
    assert result["price"] == pytest.approx(discounted_payoff.mean())
    assert result["std_error"] == pytest.approx(discounted_payoff.std(ddof=1) / math.sqrt(terminal_prices.size))


def test_monte_carlo_option_price_uses_simulated_terminal_prices():
    s0 = 100.0
    strike = 100.0
    rate = 0.05
    sigma = 0.2
    maturity = 1.0
    z_values = np.array([-1.0, 0.0, 1.0])
    rng = FixedRng(z_values)

    result = gbm.monte_carlo_option_price(
        s0,
        strike,
        rate,
        sigma,
        maturity,
        numofPaths=z_values.size,
        timeSteps=365,
        option_type="call",
        rng=rng,
    )

    terminal_prices = s0 * np.exp((rate - 0.5 * sigma**2) * maturity + sigma * np.sqrt(maturity) * z_values)
    discounted_payoff = math.exp(-rate * maturity) * np.maximum(terminal_prices - strike, 0.0)

    assert result["price"] == pytest.approx(discounted_payoff.mean())
    assert result["std_error"] == pytest.approx(discounted_payoff.std(ddof=1) / math.sqrt(z_values.size))


def test_monte_carlo_call_put_prices_use_same_terminal_draws():
    s0 = 100.0
    strike = 100.0
    rate = 0.05
    sigma = 0.2
    maturity = 1.0
    z_values = np.array([-1.0, 0.0, 1.0])
    rng = FixedRng(z_values)

    result = gbm.monte_carlo_call_put(
        s0,
        strike,
        rate,
        sigma,
        maturity,
        numofPaths=z_values.size,
        timeSteps=365,
        rng=rng,
    )

    terminal_prices = s0 * np.exp((rate - 0.5 * sigma**2) * maturity + sigma * np.sqrt(maturity) * z_values)
    discount_factor = math.exp(-rate * maturity)
    call_payoff = discount_factor * np.maximum(terminal_prices - strike, 0.0)
    put_payoff = discount_factor * np.maximum(strike - terminal_prices, 0.0)

    assert set(result) == {"call", "put"}
    assert result["call"]["price"] == pytest.approx(call_payoff.mean())
    assert result["put"]["price"] == pytest.approx(put_payoff.mean())
    assert rng.calls == [(0.0, 1.0, z_values.size)]
