# Final Year Project

Numerical option-pricing experiments for Black-Scholes models, Geometric
Brownian Motion, Monte Carlo simulation, finite-difference methods, Greeks, and
CUDA acceleration.

The project contains both Python research code and a separate CUDA console
application. The Python modules are covered by unit and integration tests.

## Project Layout

- `Bse_Explicit/`: Black-Scholes analytical pricing, explicit finite-difference
  solvers, Numba variants, and down-and-out call/put code.
- `G_Brownian_Motion/`: Brownian Motion, Geometric Brownian Motion, and Monte
  Carlo option-pricing utilities.
- `Greeks/`: Greeks-related calculations.
- `Cuda/`: CUDA implementation for Black-Scholes explicit finite-difference and
  Monte Carlo pricing.
- `tests/`: Python unit and integration tests.
- `writeup/`, `tikz/`, `plots_tex/`: dissertation, figures, and generated plot
  sources.
- `*.ipynb`: exploratory notebooks used during development.

## Python Requirements

Use Python 3.11 if possible, matching the CI workflow.

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Running Tests

Run all Python tests:

```powershell
python -m pytest
```

Run only Monte Carlo tests:

```powershell
python -m pytest tests\test_monte_carlo.py tests\integration_tests\test_monte_carlo.py
```

The GitHub Actions workflow also runs:

```powershell
python -m pytest tests -q
```

## Python Usage Examples

Monte Carlo European call pricing:

```python
from G_Brownian_Motion.Monte_Carlo import MonteCarlo

pricer = MonteCarlo(
    spot=100.0,
    strike=105.0,
    rate=0.04,
    sigma=0.25,
    maturity=1.25,
    num_paths=200_000,
)

call = pricer.call_price()
print(call["price"], call["std_error"])
```

Geometric Brownian Motion paths:

```python
from G_Brownian_Motion.Geometric_BM import geometricBrownianMotion

paths = geometricBrownianMotion(
    S0=100.0,
    mu=0.08,
    sigma=0.2,
    numofPaths=1000,
    T=1.0,
    timeSteps=365,
)
```

Explicit finite-difference call pricing:

```python
from Bse_Explicit.bse_explicit_call_put import bse_exp_call

grid, t, s, time_taken = bse_exp_call(
    sigma=0.2,
    risk_free_rate=0.05,
    time_to_exp=1.0,
    strike=100.0,
    num_steps=1000,
    spatial_step=1.0,
    spot=100.0,
)
```

## CUDA Application

The CUDA code is independent of the Python tests and notebooks. See
`Cuda/README.md` for detailed build and run instructions.

At a high level, from the `Cuda/` directory:

```powershell
cmake -S . -B build -G Ninja
cmake --build build
.\build\Cuda.exe
```

You need a working NVIDIA CUDA setup, CMake, and a compatible C++ compiler.

## Notebooks

Open notebooks with Jupyter:

```powershell
jupyter notebook
```

The notebooks are exploratory and may depend on local data, generated plots, or
the order in which cells are run. The tested Python modules should be preferred
when making code changes.

## Notes

- `Monte_Carlo.py` contains the class-based Monte Carlo pricer for European
  calls and puts.
- `Geometric_BM.py` keeps GBM path and terminal-price helpers, plus wrapper
  functions for older Monte Carlo call sites.
- Generated build outputs, figures, and LaTeX artifacts are present in the
  repository, so broad file scans can be slower than targeted commands.
