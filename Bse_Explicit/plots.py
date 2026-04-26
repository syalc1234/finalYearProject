import matplot2tikz
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt

from bse_explicit_call_put import bse_exp_call


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
    axes[0].plot(dS_list, times, marker="o", label="Python")
    axes[0].plot(dS_list, time_numba, marker="o", label="Numba")
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
def main():
    times = [287.078, 290.095, 290.604, 287.748]
    time_numba = [193.666, 189.912, 215.09, 289.426]
    dS_list = [3, 1.5, 0.75, 0.375]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    times = np.multiply(times, 1000)
    time_numba = np.multiply(time_numba, 1000)
    print("Times: ", times)
    print("Times Numba:", time_numba)
    print(dS_list)
    axes[0].plot(dS_list, times, marker="o", label="GPU Global")
    axes[0].plot(dS_list, time_numba, marker="o", label="GPU Shared")
    axes[0].legend()
    axes[0].set_xlabel(r"Spatial step $\Delta S$")
    axes[0].set_ylabel("Time taken (ms)")
    axes[0].set_title(r"Runtime vs spatial step ($\Delta S$)")
    axes[0].set_yscale('log')
    axes[0].grid(True)

    prices = np.array([4.61667158, 4.62006959, 4.62093148, 4.62114459])
    bs = float(4.6212047545501775)
    print(prices)

    axes[1].plot(dS_list, prices, marker="o", linewidth=2, label="Explicit FDM price")
    axes[1].axhline(bs, linestyle="--", linewidth=2, label="Analytical (Black-Scholes)")

    axes[1].set_xlabel(r"Spatial step size $\Delta S$")
    axes[1].set_ylabel("Option price V(S, t=0)")
    axes[1].set_title("Convergence of Explicit FDM to BS Analytical")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    matplot2tikz.save("spatialVsTimeGPU.tex", encoding="utf-8")
"""



def main():
    time_taken = np.zeros(7)

    S0 = 274.80
    K = 275.00
    T = 0.0548
    sigma = 0.1730
    r = 0.0375

    N = 25000
    dS_list = [3, 1.5, 0.75, 0.375]

    times = []
    for dS in dS_list:
        Grid, t, S, tt = bse_exp_call(sigma, r, T, K, N, dS, S0)
        times.append(float(tt))

    plt.plot(dS_list, times, marker="o", label="Python")
    plt.legend()
    plt.xlabel("Spatial step ΔS")
    plt.ylabel("Time taken (seconds)")
    plt.title("Runtime vs spatial step (ΔS)")
    plt.gca().invert_xaxis()
    plt.grid(True)
    matplot2tikz.save("../plots_tex/fdmTimeTaken.tex", encoding="utf-8")

    plt.show()




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
  ax.set_title("Explicit finite-difference Black–Scholes (Call)")

  plt.show()


if __name__ == "__main__":
    main()