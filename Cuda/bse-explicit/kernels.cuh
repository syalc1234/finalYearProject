#ifndef CUDA_KERNELS_CUH
#define CUDA_KERNELS_CUH

void calculate_coeff(float* A, float* B, float* C, float dt, float risk_free_rate, int M, float sigma);
void fill_interior_grid(float* grid, float* A, float* B, float*  C, int M, int N);
void set_terminal_condition(float* grid, float M, float strike_K, float spatial_step)

#endif //CUDA_KERNELS_CUH