

__global__ void coeff_kernel(float* A, float* B, float* C, float dt, float risk_free_rate, int M, float sigma)
{

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index + 1; i < M; i += stride )
    {
        A[i] = 0.5 * dt * ((sigma * sigma) * (i * i) - risk_free_rate * i);
        B[i] = 1 - dt * ((sigma * sigma) * (i * i)  + risk_free_rate);
        C[i] = 0.5 * dt * ((sigma * sigma) * (i * i) +  risk_free_rate * i);
    }
}

void calculate_coeff(float* A, float* B, float* C, float dt, float risk_free_rate, int M, float sigma)
{
    constexpr unsigned BLOCK_SIZE = 1024;
    const unsigned GRID_SIZE = ceil((M -1 + BLOCK_SIZE -1) / static_cast<float>(BLOCK_SIZE));
    coeff_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(A, B, C, dt, risk_free_rate, M, sigma);
}
