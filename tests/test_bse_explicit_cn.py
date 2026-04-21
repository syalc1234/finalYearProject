import math

import numpy as np

from Bse_Explicit import bse_explicit_cn as cn


def test_number_of_days_asset_level_returns_zero_float64_grid():
    grid = cn.number_of_days_asset_level(3, 4)

    assert grid.shape == (4, 5)
    assert grid.dtype == np.float64
    assert np.array_equal(grid, np.zeros((4, 5), dtype=np.float64))


def test_call_boundary_helpers_apply_expected_values():
    grid = np.full((3, 4), -1.0, dtype=np.float64)
    t = np.array([0.0, 0.5, 1.0], dtype=np.float64)
    strike = 10.0
    s_max = 15.0
    rate = 0.1
    maturity = 1.0
    spatial_step = 5.0

    grid = cn.set_boundary_condition_bottom(grid)
    grid = cn.set_terminal_condition(grid, spatial_step, strike, max_idx=2)
    grid = cn.set_right_boundary_condition(grid, strike, s_max, t, maturity, rate)

    assert np.array_equal(grid[:, 0], np.zeros(3, dtype=np.float64))
    assert np.array_equal(grid[2, :], np.array([0.0, 0.0, 0.0, 5.0]))
    assert np.allclose(
        grid[:, -1],
        np.array([s_max - strike * math.exp(-rate * (maturity - time)) for time in t]),
    )


def test_put_boundary_helpers_apply_expected_values():
    grid = np.full((3, 4), -1.0, dtype=np.float64)
    t = np.array([0.0, 0.5, 1.0], dtype=np.float64)
    strike = 10.0
    rate = 0.1
    maturity = 1.0
    spatial_step = 5.0

    grid = cn.set_terminal_condition_put(grid, spatial_step, strike, max_idx=2)
    grid = cn.set_left_boundary_condition_put(grid, strike, t, maturity, rate)
    grid = cn.set_boundary_condition_right_put(grid)

    assert np.allclose(
        grid[:, 0],
        np.array([strike * math.exp(-rate * (maturity - time)) for time in t]),
    )
    assert np.array_equal(grid[:, -1], np.zeros(3, dtype=np.float64))
    assert np.array_equal(grid[2, :], np.array([10.0, 5.0, 0.0, 0.0]))
