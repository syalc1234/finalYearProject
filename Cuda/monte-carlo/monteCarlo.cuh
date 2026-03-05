//
// Created by jayma on 21/01/2026.
//

#ifndef CUDA_MONTECARLO_CUH
#define CUDA_MONTECARLO_CUH


    void monteCarlo(float s0, float mu, float sigma, float K, int numOfPaths, int numOfSteps, float T, float r, float timeStep, float* d_normals, float*  d_s);

#endif //CUDA_MONTECARLO_CUH