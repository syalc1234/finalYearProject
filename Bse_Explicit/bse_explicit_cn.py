import time

import numpy as np


def number_of_days_asset_level(time_lvl, asset_level):
    return np.zeros((time_lvl + 1, asset_level + 1), dtype=np.float64)


def set_boundary_condition_bottom(mesh_grid):
    for i in range(mesh_grid.shape[0]):
        mesh_grid[i, 0] = 0
    return mesh_grid


def set_terminal_condition(mesh_grid, spatial_step, strike, max_idx):
    for i in range(mesh_grid.shape[1]):
        mesh_grid[max_idx, i] = max(spatial_step * i - strike, 0)
    return mesh_grid


def set_right_boundary_condition(mesh_grid, strike, s_max, t, maturity, rate):
    for k in range(mesh_grid.shape[0]):
        mesh_grid[k, -1] = s_max - strike * np.exp(-rate * (maturity - t[k]))
    return mesh_grid


def set_left_boundary_condition_put(mesh_grid, strike, t, maturity, rate):
    for k in range(mesh_grid.shape[0]):
        mesh_grid[k, 0] = strike * np.exp(-rate * (maturity - t[k]))
    return mesh_grid


def set_boundary_condition_right_put(mesh_grid):
    for i in range(mesh_grid.shape[0]):
        mesh_grid[i, -1] = 0
    return mesh_grid


def set_terminal_condition_put(mesh_grid, spatial_step, strike, max_idx):
    for i in range(mesh_grid.shape[1]):
        mesh_grid[max_idx, i] = max(strike - (spatial_step * i), 0)
    return mesh_grid


def bse_exp_call(sigma, risk_free_rate, time_to_exp, strike, num_steps, spatial_step, spot):
    start = time.time()
    s_max = max(3 * strike, 2.5 * spot)

    num_prices = int(s_max / spatial_step)
    delta_t = time_to_exp / num_steps

    t = np.linspace(0, time_to_exp, num_steps + 1)
    s = np.linspace(0, s_max, num_prices + 1)

    grid = number_of_days_asset_level(num_steps, num_prices)
    grid = set_boundary_condition_bottom(grid)
    grid = set_terminal_condition(grid, spatial_step, strike, num_steps)
    grid = set_right_boundary_condition(grid, strike, s_max, t, time_to_exp, risk_free_rate)

    a = np.zeros(num_prices + 1, dtype=np.float64)
    b = np.zeros(num_prices + 1, dtype=np.float64)
    c = np.zeros(num_prices + 1, dtype=np.float64)

    for j in range(1, num_prices):
        a[j] = 0.5 * delta_t * (sigma**2 * j**2 - risk_free_rate * j)
        b[j] = 1 - delta_t * (sigma**2 * j**2 + risk_free_rate)
        c[j] = 0.5 * delta_t * (sigma**2 * j**2 + risk_free_rate * j)

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
    s_max = max(3 * strike, 2.5 * spot)

    num_prices = int(s_max / spatial_step)
    delta_t = time_to_exp / num_steps

    t = np.linspace(0, time_to_exp, num_steps + 1)
    s = np.linspace(0, s_max, num_prices + 1)

    grid = number_of_days_asset_level(num_steps, num_prices)
    grid = set_terminal_condition_put(grid, spatial_step, strike, num_steps)
    grid = set_left_boundary_condition_put(grid, strike, t, time_to_exp, risk_free_rate)
    grid = set_boundary_condition_right_put(grid)

    a = np.zeros(num_prices + 1, dtype=np.float64)
    b = np.zeros(num_prices + 1, dtype=np.float64)
    c = np.zeros(num_prices + 1, dtype=np.float64)

    for j in range(1, num_prices):
        a[j] = 0.5 * delta_t * (sigma**2 * j**2 - risk_free_rate * j)
        b[j] = 1 - delta_t * (sigma**2 * j**2 + risk_free_rate)
        c[j] = 0.5 * delta_t * (sigma**2 * j**2 + risk_free_rate * j)

    for k in range(num_steps - 1, -1, -1):
        for i in range(1, num_prices):
            grid[k, i] = (
                a[i] * grid[k + 1, i - 1]
                + b[i] * grid[k + 1, i]
                + c[i] * grid[k + 1, i + 1]
            )

    time_taken = time.time() - start
    return grid, t, s, time_taken
