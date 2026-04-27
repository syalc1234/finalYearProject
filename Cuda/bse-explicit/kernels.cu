#include "kernels.cuh"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

void checkCuda(cudaError_t status, const char* context)
{
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(status));
    }
}

// Sets the call payoff at expiry: max(S - K, 0).
__global__ void terminalConditionKernel(float* values, int m, float strikeK, float spatialStep)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j > m) {
        return;
    }

    const float s = static_cast<float>(j) * spatialStep;
    values[j] = fmaxf(s - strikeK, 0.0f);
}

// Sets the put payoff at expiry: max(K - S, 0).
__global__ void terminalConditionPutKernel(float* values, int m, float strikeK, float spatialStep)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j > m) {
        return;
    }

    const float s = static_cast<float>(j) * spatialStep;
    values[j] = fmaxf(strikeK - s, 0.0f);
}

// Advances the explicit finite-difference grid one time step for a call.
__global__ void fillInteriorKernel(
    float* nextValues,
    const float* currentValues,
    float dt,
    float sigma,
    float sMax,
    float strikeK,
    float riskFreeRate,
    float tau,
    int m)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index > m) {
        return;
    }

    // Call lower boundary: option is worthless when S is zero.
    if (index == 0) {
        nextValues[0] = 0.0f;
        return;
    }

    // Call upper boundary: value tends to S - discounted strike.
    if (index == m) {
        nextValues[m] = sMax - strikeK * expf(-riskFreeRate * tau);
        return;
    }

    const float i = static_cast<float>(index);
    const float sigmaSquared = sigma * sigma;
    const float iSquared = i * i;
    const float a = 0.5f * dt * (sigmaSquared * iSquared - riskFreeRate * i);
    const float b = 1.0f - dt * (sigmaSquared * iSquared + riskFreeRate);
    const float c = 0.5f * dt * (sigmaSquared * iSquared + riskFreeRate * i);

    nextValues[index] = a * currentValues[index - 1]
        + b * currentValues[index]
        + c * currentValues[index + 1];
}

// Advances the explicit finite-difference grid one time step for a put.
__global__ void fillInteriorPutKernel(
    float* nextValues,
    const float* currentValues,
    float dt,
    float sigma,
    float strikeK,
    float riskFreeRate,
    float tau,
    int m)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index > m) {
        return;
    }

    // Put lower boundary: value tends to the discounted strike.
    if (index == 0) {
        nextValues[0] = strikeK * expf(-riskFreeRate * tau);
        return;
    }

    // Put upper boundary: option is worthless when S is very large.
    if (index == m) {
        nextValues[m] = 0.0f;
        return;
    }

    const float i = static_cast<float>(index);
    const float sigmaSquared = sigma * sigma;
    const float iSquared = i * i;
    const float a = 0.5f * dt * (sigmaSquared * iSquared - riskFreeRate * i);
    const float b = 1.0f - dt * (sigmaSquared * iSquared + riskFreeRate);
    const float c = 0.5f * dt * (sigmaSquared * iSquared + riskFreeRate * i);

    nextValues[index] = a * currentValues[index - 1]
        + b * currentValues[index]
        + c * currentValues[index + 1];
}

// Shared-memory variant of the time-step kernel.
__global__ void fillInteriorSharedKernel(
    float* nextValues,
    const float* currentValues,
    float dt,
    float sigma,
    float sMax,
    float strikeK,
    float riskFreeRate,
    float tau,
    int m,
    bool isCall)
{
    extern __shared__ float tile[];

    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int blockStart = blockIdx.x * blockDim.x;
    const int remaining = m - blockStart;
    const int lastActiveThread = remaining < blockDim.x - 1 ? remaining : blockDim.x - 1;
    const bool inRange = index <= m;

    // Store this block's grid values in shared memory with one-cell halo values.
    if (inRange) {
        tile[threadIdx.x + 1] = currentValues[index];
    }

    // Left halo gives thread 0 access to the previous grid point.
    if (threadIdx.x == 0) {
        const int leftIndex = blockStart > 0 ? blockStart - 1 : 0;
        tile[0] = currentValues[leftIndex];
    }

    // Right halo gives the last active thread access to the next grid point.
    if (threadIdx.x == lastActiveThread) {
        const int rightCandidate = blockStart + lastActiveThread + 1;
        const int rightIndex = rightCandidate < m ? rightCandidate : m;
        tile[lastActiveThread + 2] = currentValues[rightIndex];
    }

    __syncthreads();

    // Threads outside the grid only helped size the block; they do no pricing work.
    if (!inRange) {
        return;
    }

    if (index == 0) {
        nextValues[0] = isCall ? 0.0f : strikeK * expf(-riskFreeRate * tau);
        return;
    }

    if (index == m) {
        nextValues[m] = isCall ? sMax - strikeK * expf(-riskFreeRate * tau) : 0.0f;
        return;
    }

    const float i = static_cast<float>(index);
    const float sigmaSquared = sigma * sigma;
    const float iSquared = i * i;
    const float a = 0.5f * dt * (sigmaSquared * iSquared - riskFreeRate * i);
    const float b = 1.0f - dt * (sigmaSquared * iSquared + riskFreeRate);
    const float c = 0.5f * dt * (sigmaSquared * iSquared + riskFreeRate * i);

    nextValues[index] = a * tile[threadIdx.x]
        + b * tile[threadIdx.x + 1]
        + c * tile[threadIdx.x + 2];
}

// Reads the option value between grid nodes using linear interpolation.
float interpolatePrice(const thrust::host_vector<float>& values, float currentPrice, float spatialStep, int m)
{
    if (currentPrice <= 0.0f) {
        return values.front();
    }

    const float gridPosition = currentPrice / spatialStep;
    const int leftIndex = std::clamp(static_cast<int>(std::floor(gridPosition)), 0, m);
    const int rightIndex = std::min(leftIndex + 1, m);

    if (leftIndex == rightIndex) {
        return values[leftIndex];
    }

    const float leftPrice = static_cast<float>(leftIndex) * spatialStep;
    const float weight = (currentPrice - leftPrice) / spatialStep;
    return values[leftIndex] + weight * (values[rightIndex] - values[leftIndex]);
}

// Keeps all public BSE entry points using the same basic input validation.
void validateSettings(const optionTypeBSE& settings)
{
    if (settings.spatial_step <= 0.0f) {
        throw std::invalid_argument("spatial_step must be positive");
    }
    if (settings.N <= 0.0f) {
        throw std::invalid_argument("N must be positive");
    }
    if (settings.time_to_exp <= 0.0f) {
        throw std::invalid_argument("time_to_exp must be positive");
    }
}

// Standard/global-memory finite-difference call pricer.
BseExplicitResult priceBlackScholesExplicitCall(const optionTypeBSE& settings)
{
    validateSettings(settings);

    const float sMax = std::max(3.0f * settings.strike_K, 2.5f * settings.current_price);
    const int m = std::max(1, static_cast<int>(std::ceil(sMax / settings.spatial_step)));
    const int nSteps = std::max(1, static_cast<int>(std::round(settings.N)));
    const float dt = settings.time_to_exp / static_cast<float>(nSteps);

    constexpr int blockSize = 256;
    const int valueBlocks = std::max(1, (m + 1 + blockSize - 1) / blockSize);

    thrust::device_vector<float> currentValues(m + 1);
    thrust::device_vector<float> nextValues(m + 1);

    // Start from the payoff at maturity and step backwards to today.
    terminalConditionKernel<<<valueBlocks, blockSize>>>(
        currentValues.data().get(),
        m,
        settings.strike_K,
        settings.spatial_step);
    checkCuda(cudaGetLastError(), "terminalConditionKernel launch");

    cudaEvent_t start{};
    cudaEvent_t stop{};
    checkCuda(cudaEventCreate(&start), "cudaEventCreate(start)");
    checkCuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");
    checkCuda(cudaEventRecord(start), "cudaEventRecord(start)");

    for (int step = nSteps - 1; step >= 0; --step) {
        const float tau = settings.time_to_exp - static_cast<float>(step) * dt;

        fillInteriorKernel<<<valueBlocks, blockSize>>>(
            nextValues.data().get(),
            currentValues.data().get(),
            dt,
            settings.sigma,
            sMax,
            settings.strike_K,
            settings.risk_free_rate,
            tau,
            m);
        checkCuda(cudaGetLastError(), "fillInteriorKernel launch");

        currentValues.swap(nextValues);
    }

    checkCuda(cudaEventRecord(stop), "cudaEventRecord(stop)");
    checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");

    float timestepKernelMs = 0.0f;
    checkCuda(cudaEventElapsedTime(&timestepKernelMs, start, stop), "cudaEventElapsedTime");
    checkCuda(cudaEventDestroy(start), "cudaEventDestroy(start)");
    checkCuda(cudaEventDestroy(stop), "cudaEventDestroy(stop)");

    thrust::host_vector<float> hostValues = currentValues;
    return {interpolatePrice(hostValues, settings.current_price, settings.spatial_step, m), timestepKernelMs};
}

// Standard/global-memory finite-difference put pricer.
BseExplicitResult priceBlackScholesExplicitPut(const optionTypeBSE& settings)
{
    validateSettings(settings);

    const float sMax = std::max(3.0f * settings.strike_K, 2.5f * settings.current_price);
    const int m = std::max(1, static_cast<int>(std::ceil(sMax / settings.spatial_step)));
    const int nSteps = std::max(1, static_cast<int>(std::round(settings.N)));
    const float dt = settings.time_to_exp / static_cast<float>(nSteps);

    constexpr int blockSize = 256;
    const int valueBlocks = std::max(1, (m + 1 + blockSize - 1) / blockSize);

    thrust::device_vector<float> currentValues(m + 1);
    thrust::device_vector<float> nextValues(m + 1);

    // Start from the payoff at maturity and step backwards to today.
    terminalConditionPutKernel<<<valueBlocks, blockSize>>>(
        currentValues.data().get(),
        m,
        settings.strike_K,
        settings.spatial_step);
    checkCuda(cudaGetLastError(), "terminalConditionPutKernel launch");

    cudaEvent_t start{};
    cudaEvent_t stop{};
    checkCuda(cudaEventCreate(&start), "cudaEventCreate(start)");
    checkCuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");
    checkCuda(cudaEventRecord(start), "cudaEventRecord(start)");

    for (int step = nSteps - 1; step >= 0; --step) {
        const float tau = settings.time_to_exp - static_cast<float>(step) * dt;

        fillInteriorPutKernel<<<valueBlocks, blockSize>>>(
            nextValues.data().get(),
            currentValues.data().get(),
            dt,
            settings.sigma,
            settings.strike_K,
            settings.risk_free_rate,
            tau,
            m);
        checkCuda(cudaGetLastError(), "fillInteriorPutKernel launch");

        currentValues.swap(nextValues);
    }

    checkCuda(cudaEventRecord(stop), "cudaEventRecord(stop)");
    checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");

    float timestepKernelMs = 0.0f;
    checkCuda(cudaEventElapsedTime(&timestepKernelMs, start, stop), "cudaEventElapsedTime");
    checkCuda(cudaEventDestroy(start), "cudaEventDestroy(start)");
    checkCuda(cudaEventDestroy(stop), "cudaEventDestroy(stop)");

    thrust::host_vector<float> hostValues = currentValues;
    return {interpolatePrice(hostValues, settings.current_price, settings.spatial_step, m), timestepKernelMs};
}

// Shared-memory finite-difference call pricer.
BseExplicitResult priceBlackScholesExplicitSharedCall(const optionTypeBSE& settings)
{
    validateSettings(settings);

    const float sMax = std::max(3.0f * settings.strike_K, 2.5f * settings.current_price);
    const int m = std::max(1, static_cast<int>(std::ceil(sMax / settings.spatial_step)));
    const int nSteps = std::max(1, static_cast<int>(std::round(settings.N)));
    const float dt = settings.time_to_exp / static_cast<float>(nSteps);

    constexpr int blockSize = 512;
    const int valueBlocks = std::max(1, (m + 1 + blockSize - 1) / blockSize);
    // Each block stores its values plus one left and one right halo.
    const size_t sharedBytes = (blockSize + 2) * sizeof(float);

    thrust::device_vector<float> currentValues(m + 1);
    thrust::device_vector<float> nextValues(m + 1);

    terminalConditionKernel<<<valueBlocks, blockSize>>>(
        currentValues.data().get(),
        m,
        settings.strike_K,
        settings.spatial_step);
    checkCuda(cudaGetLastError(), "terminalConditionKernel launch");

    cudaEvent_t start{};
    cudaEvent_t stop{};
    checkCuda(cudaEventCreate(&start), "cudaEventCreate(start)");
    checkCuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");
    checkCuda(cudaEventRecord(start), "cudaEventRecord(start)");

    for (int step = nSteps - 1; step >= 0; --step) {
        const float tau = settings.time_to_exp - static_cast<float>(step) * dt;

        fillInteriorSharedKernel<<<valueBlocks, blockSize, sharedBytes>>>(
            nextValues.data().get(),
            currentValues.data().get(),
            dt,
            settings.sigma,
            sMax,
            settings.strike_K,
            settings.risk_free_rate,
            tau,
            m,
            true);
        checkCuda(cudaGetLastError(), "fillInteriorSharedKernel launch");

        currentValues.swap(nextValues);
    }

    checkCuda(cudaEventRecord(stop), "cudaEventRecord(stop)");
    checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");

    float timestepKernelMs = 0.0f;
    checkCuda(cudaEventElapsedTime(&timestepKernelMs, start, stop), "cudaEventElapsedTime");
    checkCuda(cudaEventDestroy(start), "cudaEventDestroy(start)");
    checkCuda(cudaEventDestroy(stop), "cudaEventDestroy(stop)");

    thrust::host_vector<float> hostValues = currentValues;
    return {interpolatePrice(hostValues, settings.current_price, settings.spatial_step, m), timestepKernelMs};
}

// Shared-memory finite-difference put pricer.
BseExplicitResult priceBlackScholesExplicitSharedPut(const optionTypeBSE& settings)
{
    validateSettings(settings);

    const float sMax = std::max(3.0f * settings.strike_K, 2.5f * settings.current_price);
    const int m = std::max(1, static_cast<int>(std::ceil(sMax / settings.spatial_step)));
    const int nSteps = std::max(1, static_cast<int>(std::round(settings.N)));
    const float dt = settings.time_to_exp / static_cast<float>(nSteps);

    constexpr int blockSize = 512;
    const int valueBlocks = std::max(1, (m + 1 + blockSize - 1) / blockSize);
    // Each block stores its values plus one left and one right halo.
    const size_t sharedBytes = (blockSize + 2) * sizeof(float);

    thrust::device_vector<float> currentValues(m + 1);
    thrust::device_vector<float> nextValues(m + 1);

    terminalConditionPutKernel<<<valueBlocks, blockSize>>>(
        currentValues.data().get(),
        m,
        settings.strike_K,
        settings.spatial_step);
    checkCuda(cudaGetLastError(), "terminalConditionPutKernel launch");

    cudaEvent_t start{};
    cudaEvent_t stop{};
    checkCuda(cudaEventCreate(&start), "cudaEventCreate(start)");
    checkCuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");
    checkCuda(cudaEventRecord(start), "cudaEventRecord(start)");

    for (int step = nSteps - 1; step >= 0; --step) {
        const float tau = settings.time_to_exp - static_cast<float>(step) * dt;

        fillInteriorSharedKernel<<<valueBlocks, blockSize, sharedBytes>>>(
            nextValues.data().get(),
            currentValues.data().get(),
            dt,
            settings.sigma,
            sMax,
            settings.strike_K,
            settings.risk_free_rate,
            tau,
            m,
            false);
        checkCuda(cudaGetLastError(), "fillInteriorSharedKernel launch");

        currentValues.swap(nextValues);
    }

    checkCuda(cudaEventRecord(stop), "cudaEventRecord(stop)");
    checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");

    float timestepKernelMs = 0.0f;
    checkCuda(cudaEventElapsedTime(&timestepKernelMs, start, stop), "cudaEventElapsedTime");
    checkCuda(cudaEventDestroy(start), "cudaEventDestroy(start)");
    checkCuda(cudaEventDestroy(stop), "cudaEventDestroy(stop)");

    thrust::host_vector<float> hostValues = currentValues;
    return {interpolatePrice(hostValues, settings.current_price, settings.spatial_step, m), timestepKernelMs};
}
