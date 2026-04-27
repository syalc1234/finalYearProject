import time

import numpy as np


def number_of_days_asset_level(time_lvl, asset_level):
    # Grid rows represent time levels; columns represent discretised asset prices.
    return np.zeros((time_lvl + 1, asset_level + 1), dtype=np.float64)


def set_boundary_condition_bottom(mesh_grid):
    # At S = 0, a European call is worthless at every time level.
    for i in range(mesh_grid.shape[0]):
        mesh_grid[i, 0] = 0
    return mesh_grid


def set_terminal_condition(mesh_grid, spatial_step, strike, max_idx):
    # Final row is the call payoff at expiry: max(S - K, 0).
    for i in range(mesh_grid.shape[1]):
        mesh_grid[max_idx, i] = max(spatial_step * i - strike, 0)
    return mesh_grid


def set_right_boundary_condition(mesh_grid, strike, s_max, t, maturity, rate):
    # For large S, the call behaves like S - discounted strike.
    for k in range(mesh_grid.shape[0]):
        mesh_grid[k, -1] = s_max - strike * np.exp(-rate * (maturity - t[k]))
    return mesh_grid


def set_left_boundary_condition_put(mesh_grid, strike, t, maturity, rate):
    # At S = 0, a put is worth the discounted strike.
    for k in range(mesh_grid.shape[0]):
        mesh_grid[k, 0] = strike * np.exp(-rate * (maturity - t[k]))
    return mesh_grid


def set_boundary_condition_right_put(mesh_grid):
    # For very large S, a European put is effectively worthless.
    for i in range(mesh_grid.shape[0]):
        mesh_grid[i, -1] = 0
    return mesh_grid


def set_terminal_condition_put(mesh_grid, spatial_step, strike, max_idx):
    # Final row is the put payoff at expiry: max(K - S, 0).
    for i in range(mesh_grid.shape[1]):
        mesh_grid[max_idx, i] = max(strike - (spatial_step * i), 0)
    return mesh_grid


def bse_exp_call(sigma, risk_free_rate, time_to_exp, strike, num_steps, spatial_step, spot):
    start = time.time()
    # Choose an upper asset boundary high enough to approximate infinity.
    s_max = max(3 * strike, 2.5 * spot)

    num_prices = int(s_max / spatial_step)
    delta_t = time_to_exp / num_steps

    # Build the time and asset-price axes used by the finite-difference grid.
    t = np.linspace(0, time_to_exp, num_steps + 1)
    s = np.linspace(0, s_max, num_prices + 1)

    # Apply call payoff and boundary values before stepping backwards.
    grid = number_of_days_asset_level(num_steps, num_prices)
    grid = set_boundary_condition_bottom(grid)
    grid = set_terminal_condition(grid, spatial_step, strike, num_steps)
    grid = set_right_boundary_condition(grid, strike, s_max, t, time_to_exp, risk_free_rate)

    # Explicit finite-difference coefficients for each interior asset index.
    a = np.zeros(num_prices + 1, dtype=np.float64)
    b = np.zeros(num_prices + 1, dtype=np.float64)
    c = np.zeros(num_prices + 1, dtype=np.float64)

    for j in range(1, num_prices):
        a[j] = 0.5 * delta_t * (sigma**2 * j**2 - risk_free_rate * j)
        b[j] = 1 - delta_t * (sigma**2 * j**2 + risk_free_rate)
        c[j] = 0.5 * delta_t * (sigma**2 * j**2 + risk_free_rate * j)

    # Work backwards from expiry to today using neighbouring future values.
    for k in range(num_steps - 1, -1, -1):
        for i in range(1, num_prices):
            grid[k, i] = (
                a[i] * grid[k + 1, i - 1]
                + b[i] * grid[k + 1, i]
                + c[i] * grid[k + 1, i + 1]
            )

    time_taken = time.time() - start
    return grid, t, s, time_taken


def bse_exp_put(sigma, risk_free_rate, time_to_exp, strike, num_steps, spatial_step, spot):
    start = time.time()
    # Use the same truncated asset domain as the call solver.
    s_max = max(3 * strike, 2.5 * spot)

    num_prices = int(s_max / spatial_step)
    delta_t = time_to_exp / num_steps

    # Build the time and asset-price axes used by the finite-difference grid.
    t = np.linspace(0, time_to_exp, num_steps + 1)
    s = np.linspace(0, s_max, num_prices + 1)

    # Apply put payoff and boundary values before stepping backwards.
    grid = number_of_days_asset_level(num_steps, num_prices)
    grid = set_terminal_condition_put(grid, spatial_step, strike, num_steps)
    grid = set_left_boundary_condition_put(grid, strike, t, time_to_exp, risk_free_rate)
    grid = set_boundary_condition_right_put(grid)

    # Explicit finite-difference coefficients for each interior asset index.
    a = np.zeros(num_prices + 1, dtype=np.float64)
    b = np.zeros(num_prices + 1, dtype=np.float64)
    c = np.zeros(num_prices + 1, dtype=np.float64)

    for j in range(1, num_prices):
        a[j] = 0.5 * delta_t * (sigma**2 * j**2 - risk_free_rate * j)
        b[j] = 1 - delta_t * (sigma**2 * j**2 + risk_free_rate)
        c[j] = 0.5 * delta_t * (sigma**2 * j**2 + risk_free_rate * j)

    # Work backwards from expiry to today using neighbouring future values.
    for k in range(num_steps - 1, -1, -1):
        for i in range(1, num_prices):
            grid[k, i] = (
                a[i] * grid[k + 1, i - 1]
                + b[i] * grid[k + 1, i]
                + c[i] * grid[k + 1, i + 1]
            )

    time_taken = time.time() - start
    return grid, t, s, time_taken
