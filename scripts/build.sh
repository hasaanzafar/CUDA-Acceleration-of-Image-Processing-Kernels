#!/bin/bash
set -e

echo "Building CUDA Convolution Project..."

mkdir -p build
cd build

# Compile CPU benchmark
g++ \
  ../benchmarks/benchmarks_cpu.cpp \
  ../src/cpu/convolution_cpu.cpp \
  -O2 \
  -o benchmark_cpu

# Compile CUDA benchmark
nvcc \
  ../benchmarks/benchmarks_cuda.cu \
  ../src/cuda/convolution_naive.cu \
  ../src/cuda/convolution_shared.cu \
  ../src/main.cu \
  -O2 \
  -o benchmark_cuda

echo "Build completed."
