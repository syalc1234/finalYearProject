# CUDA Option Pricing

This project is a console-based CUDA application for pricing vanilla European
options. It currently supports:

- Black-Scholes Explicit finite-difference pricing for calls and puts
- Black-Scholes Explicit shared-memory finite-difference pricing for calls and puts
- Monte Carlo pricing for calls and puts

Down-and-out/barrier options are not currently implemented or exposed in the
console menu.

## Requirements

- NVIDIA GPU with a working CUDA driver
- CUDA Toolkit
- CMake 3.22.1 or newer
- A C++20-capable host compiler
- On Windows, run CMake from a Visual Studio Developer PowerShell or Developer
  Command Prompt so `cl.exe` is available to `nvcc`

## Build

From this directory:

```powershell
cmake -S . -B build -G Ninja
cmake --build build
```

If you are building on Windows with Visual Studio instead of Ninja, configure
from a Visual Studio developer shell:

```powershell
cmake -S . -B build
cmake --build build --config Release
```

The default CUDA architecture in `CMakeLists.txt` is set to `86` for an NVIDIA
A40. Override it at configure time if you are using a different GPU:

```powershell
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=89
```

## Run

After building, run the executable from the build output directory. For example:

```powershell
.\build\Cuda.exe
```

The program first checks that a usable CUDA device is available. If no device or
driver is found, it exits before pricing.

## Console Flow

The first menu selects the pricing method:

```text
1) Black Scholes Explicit
2) Monte Carlo
```

If you select Black-Scholes Explicit, choose the implementation:

```text
1) Standard/global memory BSE
2) Shared memory BSE
```

Then choose the option type:

```text
1) Call
2) Put
```

Monte Carlo goes directly to the same call/put option type menu.

After selecting an option type, choose either:

```text
1) Enter your own values
2) Use a preset scenario
```

## Input Fields

### Black-Scholes Explicit

- `sigma`: volatility
- `risk free rate`: continuously compounded risk-free rate
- `time to expiry`: option maturity
- `K`: strike
- `spatial step`: asset-price grid spacing
- `current price`: current underlying price
- `N`: number of time steps

### Monte Carlo

- `S0`: current underlying price
- `mu`: drift
- `sigma`: volatility
- `K`: strike
- `r`: risk-free rate
- `number of paths`: simulated paths
- `number of steps`: time steps per path
- `T`: time to expiry

## Code Layout

- `main.cu`: console menus, input handling, and pricing dispatch
- `bse-explicit/`: Black-Scholes explicit finite-difference CUDA pricer
- `monte-carlo/`: CUDA Monte Carlo path simulation
- `classDef/`: parameter containers for pricing inputs

The Black-Scholes explicit code exposes both standard/global-memory functions
and shared-memory functions so the console app can benchmark or compare the two
implementations.
