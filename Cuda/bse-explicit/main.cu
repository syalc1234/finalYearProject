#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <curand.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include "bse-explicit/kernels.cuh"
#include <thrust/host_vector.h>

#include "kernels.cuh"


int main()
{
    int n = 1;
    cudaError_t e = cudaGetDeviceCount(&n);
    std::cout << "cudaGetDeviceCount: " << e
              << " (code=" << static_cast<int>(e) << "), n=" << n << "\n";

    if (e != cudaSuccess || n <= 0) {
        std::cerr << "No usable CUDA device/driver. Stopping before cuRAND calls.\n";
        return 1;
    }

   // setting the variables:
    float sigma = 0;
    float risk_free_rate = 0;
    float time_to_exp = 0;
    float strike_K = 0;
    float spatial_step = 0;
    float current_price = 0;
    float N = 0;

    float Smax = max(3 * strike_K, 2.5 * current_price);
    int M = static_cast<int>(Smax / spatial_step);
    float dt = time_to_exp / N ;

    thrust::device_vector<float> V_old(M + 1);
    thrust::device_vector<float> V_new(M + 1);

    thrust::device_vector<float> grid((N + 1) * (M + 1));

    thrust::device_vector<float> A(M);
    thrust::device_vector<float> B(M);
    thrust::device_vector<float> C(M);


    calculate_coeff(A.data().get(), B.data().get(), C.data().get(), dt, risk_free_rate, M, sigma);
    cudaDeviceSynchronize();
    set_terminal_condition(grid.data().get(), M,  strike_K,  spatial_step);

    fill_interior_grid(grid.data().get(),A.data().get(), B.data().get(), C.data().get(),M, N);
    cudaDeviceSynchronize();



    return 0;
}
