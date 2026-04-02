//
// Created by jayma on 21/01/2026.
//

#include "monteCarlo.cuh"

#include <curand.h>
#include <thrust/device_vector.h>
#include <thrust/detail/raw_pointer_cast.h>

__global__ void mc_kernel(float s0, float mu, float sigma, float K, int numOfPaths, int numOfSteps, float T, float r,
                          float timeStep, float* d_normals, float*  d_s)
{
    const unsigned pathIdx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned stride = blockDim.x * gridDim.x;
    const float discount = expf(-r * T);

    for (unsigned path = pathIdx; path < static_cast<unsigned>(numOfPaths); path += stride) {
        float s_curr = s0;

        for (int step = 0; step < numOfSteps; ++step) {
            const size_t normalIdx = static_cast<size_t>(step) * numOfPaths + path;
            s_curr += mu * s_curr * timeStep + sigma * s_curr * d_normals[normalIdx];
        }

        const float payoff = fmaxf(s_curr - K, 0.0f);
        d_s[path] = discount * payoff;
    }
}

void monteCarloLaunchKernel(float s0, float mu, float sigma, float K, int numOfPaths, int numOfSteps, float T, float r, float timeStep, float* d_normals, float*  d_s){
    const unsigned BLOCK_SIZE = 256;
    const unsigned GRID_SIZE = ceil(float(numOfPaths) / float(BLOCK_SIZE));
    mc_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(s0, mu, sigma, K, numOfPaths, numOfSteps, T, r, timeStep,  d_normals, d_s);
}

thrust::device_vector<float> monteCarlo(float s0, float mu, float sigma, float K, int numOfPaths, int numOfSteps,
                                        float T, float r)
{
    curandGenerator_t generator{};
    curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(generator, 1234ULL);

    thrust::device_vector<float> d_s(numOfPaths);

    constexpr size_t N_NORMALS = numOfPaths * numOfSteps;
    const float dt = T / static_cast<float>(numOfSteps);
    const float sqrdt = std::sqrt(dt);

    thrust::device_vector<float> d_normals(N_NORMALS);
    float* normals_ptr = thrust::raw_pointer_cast(d_normals.data());
    curandGenerateNormal(generator, normals_ptr, N_NORMALS, 0.0f, sqrdt);

    curandDestroyGenerator(generator);
    monteCarloLaunchKernel(s0, mu,sigma,K,numOfPaths,numOfSteps,T,r,dt, d_normals.data().get(),d_s.data().get());
    return d_s;
}
