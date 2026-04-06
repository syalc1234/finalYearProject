#ifndef CUDA_BSE_EXPLICIT_KERNELS_CUH
#define CUDA_BSE_EXPLICIT_KERNELS_CUH

#include "../classDef/optionTypeBSE.h"

struct BseExplicitResult
{
    float price = 0.0f;
    float timestepKernelMs = 0.0f;
};

BseExplicitResult priceBlackScholesExplicitCall(const optionTypeBSE& settings);

#endif //CUDA_BSE_EXPLICIT_KERNELS_CUH
