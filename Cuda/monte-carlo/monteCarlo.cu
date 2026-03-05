//
// Created by jayma on 21/01/2026.
//

#include "monteCarlo.cuh"

__global__ void mc_kernel(float s0, float mu, float sigma, float K, int numOfPaths, int numOfSteps, float T, float r,
 float timeStep, float* d_normals, float*  d_s)
{
    const unsigned tid = threadIdx.x;
    const unsigned bid = blockIdx.x;
    const unsigned bsz = blockDim.x;
    int s_idx = tid + bid * bsz;
    int n_idx = tid + bid * bsz;

    float s_curr = s0;

    if ( s_idx < numOfPaths )
    {
        int n = 0;
        while (n < numOfSteps)
        {
            s_curr = s_curr +  mu * s_curr * timeStep + sigma * s_curr * d_normals[n_idx];
            n_idx++;
            n++;
        }
        double payoff = s_curr - K > 0 ? s_curr - K : 0;
        __syncthreads();
        d_s[s_idx] = exp(-r * T) * payoff;

    }
    __syncthreads();
}
void monteCarlo(float s0, float mu, float sigma, float K, int numOfPaths, int numOfSteps, float T, float r, float timeStep, float* d_normals, float*  d_s){
    const unsigned BLOCK_SIZE = 1024;
    const unsigned GRID_SIZE = ceil(float(numOfPaths) / float(BLOCK_SIZE));
    monteCarlo<<<GRID_SIZE, BLOCK_SIZE>>>(s0, mu, sigma, K, numOfPaths, numOfSteps, T, r, timeStep,  d_normals, d_s);
}