cmake_minimum_required(VERSION 4.0)
project(Cuda CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(Cuda main.cu)

set_target_properties(Cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
