

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

    extern __shared__ float tile_of_grid[];
    int tx = threadIdx.x;
    int last_interior_point = M - 1;
    int tile_index = blockIdx.x * blockDim.x + threadIdx.x + 1;

    if (tile_index <= last_interior_point)
    {
        tile_of_grid[tx + 1] = V_old[tile_index];
    }

    if (tx == 0)
    {
        int left = blockIdx.x * blockDim.x;
        tile_of_grid[0] = V_old[left];
    }

    int blockLast = min((blockIdx.x + 1) * blockDim.x, last_interior_point);

    if (tile_index == blockLast)
    {
        tile_of_grid[(blockLast - blockIdx.x * blockDim.x) + 1] = V_old[blockLast];
        tile_of_grid[(blockLast - blockIdx.x * blockDim.x) + 2] = V_old[blockLast +  1];
    }

    __syncthreads();

    if ( tile_index <= last_interior_point)
    {
        V_new[tile_index] = A[tile_index] * V_old[tx]
               + B[tile_index] * V_old[tx + 1]
               + C[tile_index] * V_old[tx + 2];
    }
 }

void fill_interior_grid(float* grid, float* A, float* B, float*  C, int M, int N)
{
    constexpr int THREADS = 256;
    int blocks = (M - 1 + THREADS - 1) / THREADS;
    size_t sharedBytes = (THREADS + 2) * sizeof(float);


    for (int n = N - 1; n >= 0; --n)
    {
        float* V_new = grid +  n * (M + 1);
        float* V_old = grid + (n + 1) * (M + 1);

        fill_interior_grid_kernel<<<blocks, THREADS, sharedBytes>>>(
            V_new, V_old, A, B, C, M);
    }
}

void calculate_coeff(float* A, float* B, float* C, float dt, float risk_free_rate, int M, float sigma)
{
    constexpr unsigned BLOCK_SIZE = 1024;
    const unsigned GRID_SIZE = ceil((M -1 + BLOCK_SIZE -1) / static_cast<float>(BLOCK_SIZE));
    coeff_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(A, B, C, dt, risk_free_rate, M, sigma);
}
