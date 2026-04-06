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

__global__ void coeffKernel(float* a, float* b, float* c, float dt, float riskFreeRate, int m, float sigma)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (index >= m) {
        return;
    }

    const float sigmaSquared = sigma * sigma;
    const float i = static_cast<float>(index);
    const float iSquared = i * i;

    a[index] = 0.5f * dt * (sigmaSquared * iSquared - riskFreeRate * i);
    b[index] = 1.0f - dt * (sigmaSquared * iSquared + riskFreeRate);
    c[index] = 0.5f * dt * (sigmaSquared * iSquared + riskFreeRate * i);
}

__global__ void terminalConditionKernel(float* values, int m, float strikeK, float spatialStep)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j > m) {
        return;
    }

    const float s = static_cast<float>(j) * spatialStep;
    values[j] = fmaxf(s - strikeK, 0.0f);
}

__global__ void applyBoundaryKernel(float* values, float sMax, float strikeK, float riskFreeRate, float tau, int m)
{
    values[0] = 0.0f;
    values[m] = sMax - strikeK * expf(-riskFreeRate * tau);
}

__global__ void fillInteriorKernel(
    float* nextValues,
    const float* currentValues,
    const float* a,
    const float* b,
    const float* c,
    int m)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (index >= m) {
        return;
    }

    nextValues[index] = a[index] * currentValues[index - 1]
        + b[index] * currentValues[index]
        + c[index] * currentValues[index + 1];
}

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

float priceBlackScholesExplicitCall(const optionTypeBSE& settings)
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

    const float sMax = std::max(3.0f * settings.strike_K, 2.5f * settings.current_price);
    const int m = std::max(1, static_cast<int>(std::ceil(sMax / settings.spatial_step)));
    const int nSteps = std::max(1, static_cast<int>(std::round(settings.N)));
    const float dt = settings.time_to_exp / static_cast<float>(nSteps);

    constexpr int blockSize = 256;
    const int coeffBlocks = std::max(1, (m - 1 + blockSize - 1) / blockSize);
    const int valueBlocks = std::max(1, (m + 1 + blockSize - 1) / blockSize);

    thrust::device_vector<float> currentValues(m + 1);
    thrust::device_vector<float> nextValues(m + 1);
    thrust::device_vector<float> a(m + 1, 0.0f);
    thrust::device_vector<float> b(m + 1, 0.0f);
    thrust::device_vector<float> c(m + 1, 0.0f);

    coeffKernel<<<coeffBlocks, blockSize>>>(
        a.data().get(),
        b.data().get(),
        c.data().get(),
        dt,
        settings.risk_free_rate,
        m,
        settings.sigma);
    checkCuda(cudaGetLastError(), "coeffKernel launch");

    terminalConditionKernel<<<valueBlocks, blockSize>>>(
        currentValues.data().get(),
        m,
        settings.strike_K,
        settings.spatial_step);
    checkCuda(cudaGetLastError(), "terminalConditionKernel launch");

    for (int step = nSteps - 1; step >= 0; --step) {
        const float tau = settings.time_to_exp - static_cast<float>(step) * dt;

        applyBoundaryKernel<<<1, 1>>>(
            nextValues.data().get(),
            sMax,
            settings.strike_K,
            settings.risk_free_rate,
            tau,
            m);
        checkCuda(cudaGetLastError(), "applyBoundaryKernel launch");

        fillInteriorKernel<<<coeffBlocks, blockSize>>>(
            nextValues.data().get(),
            currentValues.data().get(),
            a.data().get(),
            b.data().get(),
            c.data().get(),
            m);
        checkCuda(cudaGetLastError(), "fillInteriorKernel launch");

        currentValues.swap(nextValues);
    }

    checkCuda(cudaDeviceSynchronize(), "BSE explicit solve");

    thrust::host_vector<float> hostValues = currentValues;
    return interpolatePrice(hostValues, settings.current_price, settings.spatial_step, m);
}
