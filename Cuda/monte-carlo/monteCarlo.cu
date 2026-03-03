//
// Created by jayma on 21/01/2026.
//

#include "monteCarlo.cuh"

__global__ void monteCarlo(float s0, float mu, float sigma, int numOfPaths, float T, float timeStep, float* d_normals)
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
        while (n < numOfPaths)
        {
            s_curr = s_curr +  mu * s_curr * timeStep + sigma * s_curr * d_normals[n_idx];
            n_idx++;
            n++;
        }
    }
    __syncthreads();

    }

}
