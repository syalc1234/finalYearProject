cmake_minimum_required(VERSION 3.22.1)
project(Cuda LANGUAGES CUDA CXX)

set(CMAKE_CUDA_FLAGS "-Wall")
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")

# A40 = SM 8.6. Keep cache-overridable for other GPUs.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

add_executable(Cuda main.cu
        monte-carlo/monteCarlo.cu
        bse-explicit/kernels.cu
        bse-explicit/kernels.cuh
        classDef/optionTypeMC.cpp
        classDef/optionTypeMC.h
        classDef/optionTypeBSE.cpp
        classDef/optionTypeBSE.h
)

target_link_libraries(Cuda PRIVATE CUDA::curand CUDA::cudart)

set_target_properties(Cuda PROPERTIES
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        CUDA_SEPARABLE_COMPILATION ON)
