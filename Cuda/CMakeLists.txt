cmake_minimum_required(VERSION 3.22.1)
project(Cuda LANGUAGES CUDA CXX)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

add_executable(Cuda main.cu
)

target_link_libraries(Cuda PRIVATE CUDA::curand CUDA::cudart)

set_target_properties(Cuda PROPERTIES
        CUDA_ARCHITECTURES native
        CUDA_SEPARABLE_COMPILATION ON)
