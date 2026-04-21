import time

import matplot2tikz
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt

from Bse_Explicit.bse_explicit_cn import set_right_boundary_condition
from Bse_Explicit.bse_explicit_cn import set_boundary_condition_bottom
from Bse_Explicit.bse_explicit_cn import number_of_days_asset_level


def bse_exp_down_out(sigma, risk_free_rate, time_to_exp, K, N, spatial_step, S0, barrier):
  start = time.time()
  Smax = max(3 * K, 2.5 * S0)

  M = int(Smax/spatial_step)
  delta_t = time_to_exp/N #delta T
  t = np.linspace(0, time_to_exp, N + 1) #Time axis N+1 points from t=0 to t=T
  S = np.linspace(barrier, Smax, M + 1) # Stock price axis M+1 points from S=0 S=Smax

  dS = (Smax - barrier)/M

  # Setting up the boundaries where the value of the option will be V(t_k, S_i)
  Grid = number_of_days_asset_level(N, M)
  Grid = set_boundary_condition_bottom(Grid)
  Grid[N, :] = np.maximum(S - K, 0.0)
  Grid[:, 0] = 0.0

  Grid = set_right_boundary_condition(Grid, K, Smax, t, time_to_exp, risk_free_rate )

  A = np.zeros(M+1, dtype=np.float64)
  B = np.zeros(M+1,dtype=np.float64)
  C = np.zeros(M+1, dtype=np.float64)
  #Calculate A_ki B_ki C_ki

  for j in range(1,M):
        S_j = barrier + j * dS
        A[j] = 0.5 * delta_t * (sigma**2 * S_j**2 / dS**2 - risk_free_rate * S_j / dS)
        B[j] = 1 - delta_t * (sigma**2 * S_j**2 / dS**2 + risk_free_rate)
        C[j] = 0.5 * delta_t * (sigma**2 * S_j**2 / dS**2 + risk_free_rate * S_j / dS)

  for k in range(N-1, -1, -1):
      for i in range(1, M):
          Grid[k, i] = (
              A[i] * Grid[k+1, i-1] +
              B[i] * Grid[k+1, i] +
              C[i] * Grid[k+1, i+1]
          )
  end = time.time()
  time_taken = end - start
  return Grid, t,S, time_taken


def _analytical_call_price(sigma, risk_free_rate, time_to_exp, K, S0):
  d1 = (
      np.log(S0 / K) + (risk_free_rate + 0.5 * sigma**2) * time_to_exp
  ) / (sigma * np.sqrt(time_to_exp))
  d2 = d1 - sigma * np.sqrt(time_to_exp)

  return float(
      S0 * sp.stats.norm.cdf(d1)
      - K * np.exp(-risk_free_rate * time_to_exp) * sp.stats.norm.cdf(d2)
  )


def show_barrier_convergence(
    sigma,
    risk_free_rate,
    time_to_exp,
    K,
    N,
    spatial_step,
    S0,
    barriers=None,
    plot=True,
    print_table=True,
):
  """Show down-and-out call convergence to the vanilla call as H moves to 0.

  The barrier is moved down and away from the spot. As H approaches 0, the
  down-and-out option should converge to the vanilla Black-Scholes call price.
  """
  if barriers is None:
    barriers = np.array(
        [
            0.98 * S0,
            0.95 * S0,
            0.90 * S0,
            0.80 * S0,
            0.65 * S0,
            0.50 * S0,
            0.25 * S0,
            0.0,
        ],
        dtype=np.float64,
    )
  else:
    barriers = np.asarray(barriers, dtype=np.float64)

  if np.any(barriers < 0):
    raise ValueError("barriers must be non-negative for a down-and-out call")
  if np.any(barriers > S0):
    raise ValueError("each barrier must be at or below S0")

  analytical_price = _analytical_call_price(
      sigma, risk_free_rate, time_to_exp, K, S0
  )
  rows = []

  for barrier in barriers:
    Grid, _, S, time_taken = bse_exp_down_out(
        sigma,
        risk_free_rate,
        time_to_exp,
        K,
        N,
        spatial_step,
        S0,
        float(barrier),
    )
    spot_price = float(sp.interpolate.CubicSpline(S, Grid[0, :])(S0))
    absolute_error = abs(spot_price - analytical_price)
    relative_error = (
        absolute_error / abs(analytical_price)
        if analytical_price != 0.0
        else np.nan
    )

    rows.append(
        {
            "barrier": float(barrier),
            "distance_from_spot": float(S0 - barrier),
            "down_out_price": spot_price,
            "analytical_call_price": analytical_price,
            "absolute_error": absolute_error,
            "relative_error": relative_error,
            "time_taken": float(time_taken),
        }
    )

  if print_table:
    print("Barrier convergence to vanilla Black-Scholes call")
    print(f"Analytical call price: {analytical_price:.8f}")
    print(
        f"{'Barrier':>12} {'S0-H':>12} {'Down-out':>14} "
        f"{'Abs error':>14} {'Rel error':>12}"
    )
    for row in rows:
      print(
          f"{row['barrier']:12.4f} "
          f"{row['distance_from_spot']:12.4f} "
          f"{row['down_out_price']:14.8f} "
          f"{row['absolute_error']:14.8f} "
          f"{row['relative_error']:12.4%}"
      )

  if plot:
    distances = np.array([row["distance_from_spot"] for row in rows])
    down_out_prices = np.array([row["down_out_price"] for row in rows])
    absolute_errors = np.array([row["absolute_error"] for row in rows])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(distances, down_out_prices, marker="o", label="Down-and-out call")
    axes[0].axhline(
        analytical_price,
        linestyle="--",
        linewidth=2,
        label="Analytical vanilla call",
    )
    axes[0].set_xlabel("Distance from spot to barrier (S0 - H)")
    axes[0].set_ylabel("Option price")
    axes[0].set_title("Barrier moving away")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].semilogy(distances, absolute_errors, marker="o")
    axes[1].set_xlabel("Distance from spot to barrier (S0 - H)")
    axes[1].set_ylabel("Absolute error")
    axes[1].set_title("Error against vanilla call")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    matplot2tikz.save("convergenceBarrier", encoding="utf-8")

    plt.show()

  return rows


def main():
    """
    rows = show_barrier_convergence(
        sigma=0.4030,
        risk_free_rate=0.0475,
        time_to_exp=0.0648,
        K=250,
        N=25000,
        spatial_step=0.75,
        S0=245,
        barriers=[270, 265, 260, 255, 250, 245  ],
    )
"""
    sigma = 0.1730
    risk_free_rate = 0.0375
    time_to_exp = 0.0548
    K = 275.00
    N = 25000
    spatial_step = 0.75
    S0 = 274.80
    barrier = 270

    Grid, T, S, t = bse_exp_down_out(sigma, risk_free_rate, time_to_exp, K, N, spatial_step, S0, 270)
    spl = sp.interpolate.CubicSpline(S, Grid[0, :])

    print("SPL", float(spl(274.80)))
    plot_bse_down_out(Grid, T, S)

def plot_bse_down_out(Grid, T, S):
  T_grid, S_grid = np.meshgrid(T, S, indexing="ij")
  fig = plt.figure(figsize=(10, 6))
  ax = fig.add_subplot(111, projection="3d")

  surface = ax.plot_surface(
      S_grid,    # x-axis: underlying price
      T_grid,    # y-axis: time
      Grid,      # z-axis: option value
      cmap="rainbow",
      linewidth=0,
      antialiased=True
  )

  ax.set_xlabel("Underlying price S")
  ax.set_ylabel("Time t")
  ax.set_zlabel("Option value V(t, S)")
  ax.set_title("Explicit finite-difference Black–Scholes (Call)")
  ax.view_init(elev=25, azim=-135)

  plt.show()

if __name__ == "__main__":
    main()
