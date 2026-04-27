import matplot2tikz
import numpy as np
from matplotlib import pyplot as plt

def gpu_main():
    times = [2.85850041e+10, 5.33451560e+10, 1.16518953e+11, 2.73269999e+11]
    time_numba = [2.05644131e+09, 6.81110382e+08, 5.68627596e+08, 1.13910413e+09]
    dS_list = [3, 1.5, 0.75, 0.375]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    times = np.multiply(times, 1000)
    time_numba = np.multiply(time_numba, 1000)
    print("Times: ", times)
    print("Times Numba:", time_numba)
    print(dS_list)
    axes[0].plot(dS_list, times, marker="o", label="GPU")
    axes[0].plot(dS_list, time_numba, marker="o", label="GPU Shared")
    axes[0].legend()
    axes[0].set_xlabel(r"Spatial step $\Delta S$")
    axes[0].set_ylabel("Time taken (ms)")
    axes[0].set_title(r"Runtime vs spatial step ($\Delta S$)")
    axes[0].set_yscale('log')
    axes[0].grid(True)

    prices = np.array([4.61667158487682, 4.62006959414754, 4.6209314845475, 4.62114458687641])
    bs = float(4.6212047545501775)
    print(prices)

    axes[1].plot(dS_list, prices, marker="o", linewidth=2, label="Explicit FDM price")
    axes[1].axhline(bs, linestyle="--", linewidth=2, label="Analytical (Black-Scholes)")

    axes[1].set_xlabel(r"Spatial step size $\Delta S$")
    axes[1].set_ylabel("Option price V(S, t=0)")
    axes[1].set_title("Convergence of Explicit FDM to BS Analytical")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    matplot2tikz.save("spatialVsTime.tex", encoding="utf-8")

""" 
def main
"""


def plot_bse(Grid, T, S):
  T_grid, S_grid = np.meshgrid(T, S, indexing="ij")
  fig = plt.figure(figsize=(10, 6))
  ax = fig.add_subplot(111, projection="3d")

  ax.plot_surface(
      S_grid,    # x-axis: underlying price
      T_grid,    # y-axis: time
      Grid,      # z-axis: option value
      linewidth=0,
      antialiased=True
  )

  ax.set_xlabel("Underlying price S")
  ax.set_ylabel("Time t")
  ax.set_zlabel("Option value V(t, S)")
  ax.set_title("Explicit finite-difference Black–Scholes")

  plt.show()