cmake_minimum_required(VERSION 3.18)
project(cuda_convolution LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

add_executable(benchmark_cpu
    benchmarks/benchmarks_cpu.cpp
    src/cpu/convolution_cpu.cpp
)

add_executable(benchmark_cuda
    benchmarks/benchmarks_cuda.cu
    src/cuda/convolution_naive.cu
    src/cuda/convolution_shared.cu
    src/main.cu
)

set_target_properties(benchmark_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
