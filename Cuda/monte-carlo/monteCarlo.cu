//
// Created by jayma on 21/01/2026.
//

#include "monteCarlo.cuh"

#include <cmath>
#include <curand.h>
#include <thrust/device_vector.h>
#include <thrust/detail/raw_pointer_cast.h>

__global__ void mc_kernel(float s0, float sigma, float K, int numOfPaths, float T, float r,
                          const float* d_normals, float* d_s)
{
    const unsigned pathIdx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned stride = blockDim.x * gridDim.x;
    const float discount = expf(-r * T);
    const float drift = (r - 0.5f * sigma * sigma) * T;
    const float sqrtT = sqrtf(T);

    for (unsigned path = pathIdx; path < static_cast<unsigned>(numOfPaths); path += stride) {
        // Match the Python reference: exact GBM terminal-price sampling under risk-neutral drift.
        const float z = d_normals[path];
        const float diffusion = sigma * sqrtT * z;
        const float s_terminal = s0 * expf(drift + diffusion);
        const float payoff = fmaxf(s_terminal - K, 0.0f);
        d_s[path] = discount * payoff;
    }
}

void monteCarloLaunchKernel(float s0, float sigma, float K, int numOfPaths, float T, float r, float* d_normals, float* d_s)
{
    const unsigned BLOCK_SIZE = 256;
    const unsigned GRID_SIZE = ceil(float(numOfPaths) / float(BLOCK_SIZE));
    mc_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(s0, sigma, K, numOfPaths, T, r, d_normals, d_s);
}

thrust::device_vector<float> monteCarlo(float s0, float mu, float sigma, float K, int numOfPaths, int numOfSteps,
                                        float T, float r)
{
    static_cast<void>(mu);
    static_cast<void>(numOfSteps);

    curandGenerator_t generator{};
    curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(generator, 1234ULL);

    thrust::device_vector<float> d_s(numOfPaths);

    thrust::device_vector<float> d_normals(numOfPaths);
    float* normals_ptr = thrust::raw_pointer_cast(d_normals.data());
    curandGenerateNormal(generator, normals_ptr, numOfPaths, 0.0f, 1.0f);

    curandDestroyGenerator(generator);
    monteCarloLaunchKernel(s0, sigma, K, numOfPaths, T, r, d_normals.data().get(), d_s.data().get());
    return d_s;
}
