import numpy as np
import matplotlib.pyplot as plt
import matplot2tikz

try:
    from Bse_Explicit.bse_explicit_call_put import bse_exp_call, bse_exp_put
except ImportError:
    from Bse_Explicit.bse_explicit_call_put import bse_exp_call, bse_exp_put


def calculate_delta_gamma_theta(grid,dt, S):
    V0 = grid[0, :]
    V1 = grid[1, :]

    delta = np.gradient(V0, S)
    gamma = np.gradient(delta, S)
    theta = (V1 - V0) / dt

    return delta, theta, gamma


def plot_greeks(delta,delta_, theta, theta_, gamma,gamma_, S, K):
    plt.figure(figsize=(8, 5))
    plt.plot(S, delta, label="Call Option")
    plt.plot(S, delta_, label="Put Option")
    plt.xlabel("Stock Price (S)")
    plt.ylabel("Delta")
    plt.title(r'Impact of Stock Price(S) on Delta ($\Delta$)''\n'
              )
    plt.axvline(K, color='black', linestyle='dashed', linewidth=2, label="Strike ")
    plt.legend()
    plt.grid(True)
   # matplot2tikz.save("../Plots/Delta.tex")
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(S, gamma, label="Call & Put Option")
    plt.plot(S, gamma_, label="Call & Put Option")
    plt.xlabel("Stock Price (S)")
    plt.ylabel("Gamma")
    plt.title(r'Impact of Stock Price(S) on Gamma ($\Gamma$)')
    plt.axvline(K, color='black', linestyle='dashed', linewidth=2, label="Strike")
    plt.legend()
    plt.grid(True)
  #  matplot2tikz.save("../Plots/Gamma.tex")
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(S, theta, label="Call Option")
    plt.plot(S, theta_, label="Put Option")
    plt.xlabel("Stock Price (S)")
    plt.ylabel("Theta")
    plt.title(r'Impact of Stock Price(S) on Theta ($\Theta$)')
    plt.axvline(K, color='black', linestyle='dashed', linewidth=2, label="Strike")
    plt.legend()
    plt.grid(True)
   # matplot2tikz.save("../Plots/Theta.tex")
    plt.show()


def main():
    sigma = 0.2
    r = 0.05
    maturity = 1.0
    K = 100
    S0 = 100

    N = 25000
    spatial_step = 1.0
    delta_t = maturity / N

    Grid, time_grid, S, call_runtime = bse_exp_call(sigma, r, maturity, K, N, spatial_step, S0)
    Grid_, time_grid_put, S_, put_runtime = bse_exp_put(sigma, r, maturity, K, N, spatial_step, S0)

    #Call
    delta, theta, gamma  = calculate_delta_gamma_theta(Grid, delta_t, S)
    #Put
    delta_, theta_, gamma_  = calculate_delta_gamma_theta(Grid_, delta_t, S_)
    plot_greeks(delta,delta_, theta,theta_, gamma,gamma_, S,K)



if __name__ == '__main__':
    # execute only if run as the entry point into the program
    main()
