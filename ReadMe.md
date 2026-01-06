# Final Year Project

Numerical and analytical experiments for option pricing and stochastic processes.
The notebooks focus on the Black–Scholes equation (BSE) using finite-difference
schemes, alongside analytical pricing.

## Contents
- `analytical.ipynb`: Analytical Black–Scholes pricing and implied volatility.
- `BSE_Explicit_CN.ipynb`: Explicit and Crank–Nicolson finite-difference schemes.
- `BrownianMotion.ipynb`: Brownian motion and Geometric Brownian Motion simulations and visualizations.
- `Cuda/`: CUDA implementation and build files.

## Requirements
Activate Venv:
```bash
.venv/Scripts/activate
```
Install dependencies:

```bash
pip -m pip install -r requirements.txt
```

## Usage
Open the notebooks with Jupyter:

```bash
jupyter notebook
```

Then run the desired notebook in order.

## Notes
- The CUDA code lives in `Cuda/` and is independent of the Python notebooks.
- If you clone into a new environment, ensure Jupyter is available in your
  Python environment.
