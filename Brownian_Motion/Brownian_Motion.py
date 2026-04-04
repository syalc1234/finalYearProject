import matplot2tikz
import numpy as np
from matplotlib import pyplot as plt


def geometricBrownianMotion(S0, mu, sigma, numofPaths, T, timeSteps):
    dt = T/timeSteps

    # Browniam increments
    dW = np.random.normal(0, np.sqrt(dt), size=(numofPaths, timeSteps)).T

    #Calculation for each step
    drift = (mu - ((sigma**2)/2)) * dt
    diffusion = sigma * dW
    increments = np.exp(drift + diffusion)


    St = np.vstack([np.ones(numofPaths), increments]).cumprod(axis=0) * S0

    return St


gBMSim = geometricBrownianMotion(100, 0.2, 0.1, 30, 1.0, 365)
plt.plot(gBMSim)
plt.xlabel("Time (Days)")
plt.ylabel("Price ($)")
plt.title(r'Geometric Brownian Motion with 75 paths, with $\sigma = 0.1$ and $\mu$=0.2')
matplot2tikz.save("GBM")
plt.show()

def main():
    gBMSim = geometricBrownianMotion(100, 0.2, 0.1, 30, 1.0, 365)
    plt.plot(gBMSim)
    plt.xlabel("Time (Days)")
    plt.ylabel("Price ($)")
    plt.title(r'Geometric Brownian Motion with 75 paths, with $\sigma = 0.1$ and $\mu$=0.2')
    matplot2tikz.save("GBM")
    plt.show()

if __name__ == '__main__':
    # execute only if run as the entry point into the program
    main()
