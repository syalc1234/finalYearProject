#include <cmath>
#include <cuda_runtime.h>
#include <exception>
#include <iostream>
#include <limits>
#include <string>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>

#include "bse-explicit/kernels.cuh"
#include "classDef/optionTypeBSE.h"
#include "classDef/optionTypeMC.h"
#include "monte-carlo/monteCarlo.cuh"

namespace
{
struct PricingResult
{
    double price = 0.0;
    float elapsedMs = 0.0f;
};

void clearInput()
{
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
}

template <typename T>
T readValue(const std::string& prompt)
{
    T value{};

    while (true) {
        std::cout << prompt;
        if (std::cin >> value) {
            return value;
        }

        std::cout << "Invalid input. Try again.\n";
        clearInput();
    }
}

int readMenuChoice(const std::string& prompt, int minChoice, int maxChoice)
{
    while (true) {
        const int choice = readValue<int>(prompt);
        if (choice >= minChoice && choice <= maxChoice) {
            return choice;
        }

        std::cout << "Please choose a value between " << minChoice << " and " << maxChoice << ".\n";
    }
}

// Built-in Monte Carlo scenarios used when the user selects a preset.
optionTypeMC getMonteCarloPreset(int optionType)
{
    switch (optionType) {
        case 1:
            return {274.80f, 0.0375f, 0.1730f, 275.00f, 0.0375f, 33554432, 128, 1.0f};
        case 2:
            return {100.0f, 0.2f, 0.25f, 105.0f, 0.05f, 100000, 1000, 1.0f};
        default:
            return {};
    }
}

// Built-in finite-difference scenarios used when the user selects a preset.
optionTypeBSE getBsePreset(int optionType)
{
    switch (optionType) {
        case 1:
            return {0.1730f, 0.0375f, 0.0548f, 275.00f, 0.375f, 274.80f, 100000.0f};
        case 2:
            return {0.25f, 0.05f, 1.0f, 105.0f, 1.0f, 100.0f, 1000.0f};
        default:
            return {};
    }
}

// Reads either preset or custom inputs for the Monte Carlo pricer.
optionTypeMC readMonteCarloSettings(int optionType)
{
    std::cout << "\n1) Enter your own values\n";
    std::cout << "2) Use a preset scenario\n";
    const int sourceChoice = readMenuChoice("Choose how to set the Monte Carlo inputs: ", 1, 2);

    if (sourceChoice == 2) {
        std::cout << "Using preset values. Update `getMonteCarloPreset()` in main.cu to add more presets.\n";
        return getMonteCarloPreset(optionType);
    }

    const auto s0 = readValue<float>("Spot price (S0): ");
    const auto mu = readValue<float>("Drift (mu): ");
    const auto sigma = readValue<float>("Volatility (sigma): ");
    const auto strikeK = readValue<float>("Strike (K): ");
    const auto r = readValue<float>("Risk free rate (r): ");
    const auto numberPaths = readValue<float>("Number of paths: ");
    const auto numberSteps = readValue<float>("Number of steps: ");
    const auto timeToExpiry = readValue<float>("Time to expiry (T): ");

    return {s0, mu, sigma, strikeK, r, numberPaths, numberSteps, timeToExpiry};
}

// Reads either preset or custom inputs for the Black-Scholes explicit pricer.
optionTypeBSE readBseSettings(int optionType)
{
    std::cout << "\n1) Enter your own values\n";
    std::cout << "2) Use a preset scenario\n";
    const int sourceChoice = readMenuChoice("Choose how to set the Black-Scholes Explicit inputs: ", 1, 2);

    if (sourceChoice == 2) {
        std::cout << "Using preset values. Update `getBsePreset()` in main.cu to add more presets.\n";
        return getBsePreset(optionType);
    }

    const auto sigma = readValue<float>("Volatility (sigma): ");
    const auto riskFreeRate = readValue<float>("Risk free rate: ");
    const auto timeToExp = readValue<float>("Time to expiry: ");
    const auto strikeK = readValue<float>("Strike (K): ");
    const auto spatialStep = readValue<float>("Spatial step: ");
    const auto currentPrice = readValue<float>("Current price: ");
    const auto n = readValue<float>("Number of time steps (N): ");

    return {sigma, riskFreeRate, timeToExp, strikeK, spatialStep, currentPrice, n};
}

// Runs the CUDA Monte Carlo call simulation and averages discounted payoffs.
PricingResult runMonteCarloCall(const optionTypeMC& settings)
{
    cudaEvent_t start{};
    cudaEvent_t stop{};
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    thrust::device_vector<float> discountedPayoffs = monteCarlo(
        settings.s0,
        settings.mu,
        settings.sigma,
        settings.K,
        settings.Number_Paths,
        settings.Number_Steps,
        settings.T,
        settings.r);

    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsedMs = 0.0f;
    cudaEventElapsedTime(&elapsedMs, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    const float payoffSum = thrust::reduce(discountedPayoffs.begin(), discountedPayoffs.end(), 0.0f);
    const double optionPrice = static_cast<double>(payoffSum) / static_cast<double>(settings.Number_Paths);

    return {optionPrice, elapsedMs};
}

// Uses put-call parity to derive the put from the Monte Carlo call estimate.
PricingResult runMonteCarloPut(const optionTypeMC& settings)
{
    PricingResult result = runMonteCarloCall(settings);
    result.price = result.price - static_cast<double>(settings.s0)
        + static_cast<double>(settings.K) * std::exp(-static_cast<double>(settings.r) * static_cast<double>(settings.T));
    return result;
}

// Runs the CUDA explicit finite-difference call pricer.
void runBlackScholesExplicitCall(const optionTypeBSE& settings, bool useSharedMemory)
{
    std::cout << (useSharedMemory
        ? "\nBlack-Scholes Explicit Shared-Memory Call selected.\n"
        : "\nBlack-Scholes Explicit Standard Call selected.\n");
    std::cout << "Current preset/custom settings:\n";
    std::cout << "sigma=" << settings.sigma
              << ", r=" << settings.risk_free_rate
              << ", T=" << settings.time_to_exp
              << ", K=" << settings.strike_K
              << ", dS=" << settings.spatial_step
              << ", S=" << settings.current_price
              << ", N=" << settings.N << '\n';

    try {
        const BseExplicitResult result = useSharedMemory
            ? priceBlackScholesExplicitSharedCall(settings)
            : priceBlackScholesExplicitCall(settings);
        std::cout << "Option Price: " << result.price << '\n';
        std::cout << "Time-Step Kernel Loop (ms): " << result.timestepKernelMs << '\n';
    } catch (const std::exception& ex) {
        std::cerr << "Black-Scholes explicit solver failed: " << ex.what() << '\n';
    }
}

// Runs the CUDA explicit finite-difference put pricer.
void runBlackScholesExplicitPut(const optionTypeBSE& settings, bool useSharedMemory)
{
    std::cout << (useSharedMemory
        ? "\nBlack-Scholes Explicit Shared-Memory Put selected.\n"
        : "\nBlack-Scholes Explicit Standard Put selected.\n");
    std::cout << "Current preset/custom settings:\n";
    std::cout << "sigma=" << settings.sigma
              << ", r=" << settings.risk_free_rate
              << ", T=" << settings.time_to_exp
              << ", K=" << settings.strike_K
              << ", dS=" << settings.spatial_step
              << ", S=" << settings.current_price
              << ", N=" << settings.N << '\n';

    try {
        const BseExplicitResult result = useSharedMemory
            ? priceBlackScholesExplicitSharedPut(settings)
            : priceBlackScholesExplicitPut(settings);
        std::cout << "Option Price: " << result.price << '\n';
        std::cout << "Time-Step Kernel Loop (ms): " << result.timestepKernelMs << '\n';
    } catch (const std::exception& ex) {
        std::cerr << "Black-Scholes explicit solver failed: " << ex.what() << '\n';
    }
}

// Sends the selected finite-difference option type to the correct pricer.
void dispatchBlackScholesExplicit(int implementationType, int optionType, const optionTypeBSE& settings)
{
    // Menu option 2 selects the shared-memory BSE implementation.
    const bool useSharedMemory = implementationType == 2;

    switch (optionType) {
        case 1:
            runBlackScholesExplicitCall(settings, useSharedMemory);
            break;
        case 2:
            runBlackScholesExplicitPut(settings, useSharedMemory);
            break;
        default:
            std::cout << "Unsupported Black-Scholes Explicit option selected.\n";
            break;
    }
}

// Sends the selected Monte Carlo option type to the correct pricer.
void dispatchMonteCarlo(int optionType, const optionTypeMC& settings)
{
    PricingResult result{};

    switch (optionType) {
        case 1:
            result = runMonteCarloCall(settings);
            std::cout << "\nMonte Carlo Call selected.\n";
            break;
        case 2:
            result = runMonteCarloPut(settings);
            std::cout << "\nMonte Carlo Put selected.\n";
            break;
        default:
            std::cout << "Unsupported Monte Carlo option selected.\n";
            return;
    }

    std::cout << "Option Price: " << result.price << '\n';
    std::cout << "Elapsed Time (ms): " << result.elapsedMs << '\n';
}
} // namespace

int main()
{
    int n = 1;
    cudaError_t e = cudaGetDeviceCount(&n);
    std::cout << "cudaGetDeviceCount: " << e
              << " (code=" << static_cast<int>(e) << "), n=" << n << "\n";

    if (e != cudaSuccess || n <= 0) {
        std::cerr << "No usable CUDA device/driver. Stopping before option pricing.\n";
        return 1;
    }

    std::cout << R"(
                      /$$$$$$  /$$$$$$$  /$$   /$$        /$$$$$$  /$$$$$$$  /$$$$$$$$ /$$$$$$  /$$$$$$  /$$   /$$       /$$$$$$$  /$$$$$$$  /$$$$$$  /$$$$$$  /$$$$$$ /$$   /$$  /$$$$$$
                     /$$__  $$| $$__  $$| $$  | $$       /$$__  $$| $$__  $$|__  $$__/|_  $$_/ /$$__  $$| $$$ | $$      | $$__  $$| $$__  $$|_  $$_/ /$$__  $$|_  $$_/| $$$ | $$ /$$__  $$
                    | $$  \__/| $$  \ $$| $$  | $$      | $$  \ $$| $$  \ $$   | $$     | $$  | $$  \ $$| $$$$| $$      | $$  \ $$| $$  \ $$  | $$  | $$  \__/  | $$  | $$$$| $$| $$  \__/
                    | $$ /$$$$| $$$$$$$/| $$  | $$      | $$  | $$| $$$$$$$/   | $$     | $$  | $$  | $$| $$ $$ $$      | $$$$$$$/| $$$$$$$/  | $$  | $$        | $$  | $$ $$ $$| $$ /$$$$
                    | $$|_  $$| $$____/ | $$  | $$      | $$  | $$| $$____/    | $$     | $$  | $$  | $$| $$  $$$$      | $$____/ | $$__  $$  | $$  | $$        | $$  | $$  $$$$| $$|_  $$
                    | $$  \ $$| $$      | $$  | $$      | $$  | $$| $$         | $$     | $$  | $$  | $$| $$\  $$$      | $$      | $$  \ $$  | $$  | $$    $$  | $$  | $$\  $$$| $$  \ $$
                    |  $$$$$$/| $$      |  $$$$$$/      |  $$$$$$/| $$         | $$    /$$$$$$|  $$$$$$/| $$ \  $$      | $$      | $$  | $$ /$$$$$$|  $$$$$$/ /$$$$$$| $$ \  $$|  $$$$$$/
                     \______/ |__/        \______/        \______/ |__/         |__/   |______/ \______/ |__/  \__/      |__/      |__/  |__/|______/ \______/ |______/|__/  \__/ \______/
    )" << '\n';

    std::cout << "1) Black Scholes Explicit\n";
    std::cout << "2) Monte Carlo\n";
    const int pricingMethod = readMenuChoice("Choose a pricing method: ", 1, 2);

    if (pricingMethod == 1) {
        // Choose the finite-difference implementation before choosing call/put.
        std::cout << "\n1) Standard/global memory BSE\n";
        std::cout << "2) Shared memory BSE\n";
        const int implementationType = readMenuChoice("Choose a Black-Scholes implementation: ", 1, 2);

        std::cout << "\n1) Call\n";
        std::cout << "2) Put\n";
        const int optionType = readMenuChoice("Choose an option type: ", 1, 2);
        const optionTypeBSE settings = readBseSettings(optionType);
        dispatchBlackScholesExplicit(implementationType, optionType, settings);
    } else {
        std::cout << "\n1) Call\n";
        std::cout << "2) Put\n";
        const int optionType = readMenuChoice("Choose an option type: ", 1, 2);
        const optionTypeMC settings = readMonteCarloSettings(optionType);
        dispatchMonteCarlo(optionType, settings);
    }

    return 0;
}
