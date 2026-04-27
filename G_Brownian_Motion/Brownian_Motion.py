import numpy as np


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