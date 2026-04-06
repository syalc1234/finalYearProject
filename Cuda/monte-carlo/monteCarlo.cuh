//
// Created by jayma on 21/01/2026.
//

#ifndef CUDA_MONTECARLO_CUH
#define CUDA_MONTECARLO_CUH
#include <thrust/device_vector.h>


void monteCarloLaunchKernel(float s0, float sigma, float K, int numOfPaths, float T, float r, float* d_normals, float* d_s);
thrust::device_vector<float> monteCarlo(float s0, float mu, float sigma, float K, int numOfPaths, int numOfSteps,
                                        float T, float r);
#endif //CUDA_MONTECARLO_CUH
