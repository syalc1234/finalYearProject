

__global__ void coeff_kernel(float* A, float* B, float* C, float dt, float risk_free_rate, int M, float sigma)
{

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float sigma_sqrd = sigma * sigma;

    for (int i = index + 1; i < M; i += stride )
    {
        float i_sqrd = static_cast<float>(i*i);

        A[i] = 0.5f * dt * (sigma_sqrd * i_sqrd - risk_free_rate * i);
        B[i] = 1.0f - dt * (sigma_sqrd * i_sqrd  + risk_free_rate);
        C[i] = 0.5f * dt * (sigma_sqrd * i_sqrd +  risk_free_rate * i);
    }
}

__global__ void fill_interior_grid_kernel(float* V_new, float* V_old, float* A, float* B, float* C, int M)
{

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < M; i += stride )
    {
        V_new[i] = A[i] * V_old[i - 1]
                 + B[i] * V_old[i]
                 + C[i] * V_old[i + 1];
    }
 }

void fill_interior_grid(float* grid, float* A, float* B, float*  C, int M, int N)
{
    constexpr int THREADS = 256;
    int blocks = (M - 1 + THREADS - 1) / THREADS;

    for (int n = N - 1; n >= 0; --n)
    {
        float* V_new = grid +  n * (M + 1);
        float* V_old = grid + (n + 1) * (M + 1);

        fill_interior_grid_kernel<<<blocks, THREADS>>>(
            V_new, V_old, A, B, C, M);
    }
}

void calculate_coeff(float* A, float* B, float* C, float dt, float risk_free_rate, int M, float sigma)
{
    constexpr unsigned BLOCK_SIZE = 1024;
    const unsigned GRID_SIZE = ceil((M -1 + BLOCK_SIZE -1) / static_cast<float>(BLOCK_SIZE));
    coeff_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(A, B, C, dt, risk_free_rate, M, sigma);
}
