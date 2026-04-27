#ifndef CUDA_BSE_EXPLICIT_KERNELS_CUH
#define CUDA_BSE_EXPLICIT_KERNELS_CUH

#include "../classDef/optionTypeBSE.h"

struct BseExplicitResult
{
    float price = 0.0f;
    float timestepKernelMs = 0.0f;
};

BseExplicitResult priceBlackScholesExplicitCall(const optionTypeBSE& settings);
BseExplicitResult priceBlackScholesExplicitPut(const optionTypeBSE& settings);

// Shared-memory variants use the same finite-difference scheme but cache
// neighbouring grid values inside each CUDA block.
BseExplicitResult priceBlackScholesExplicitSharedCall(const optionTypeBSE& settings);
BseExplicitResult priceBlackScholesExplicitSharedPut(const optionTypeBSE& settings);

#endif //CUDA_BSE_EXPLICIT_KERNELS_CUH
