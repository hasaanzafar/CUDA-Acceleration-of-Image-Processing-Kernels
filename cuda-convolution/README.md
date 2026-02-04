# CUDA Convolution Acceleration

This project implements and benchmarks custom CUDA kernels for 2D image convolution,
demonstrating GPU parallelism and memory optimization techniques.

## Implementations
- Naive CUDA convolution (global memory)
- Shared memory optimized convolution (tiling)
- CPU baseline implementation

## Techniques Demonstrated
- CUDA thread hierarchy (grid, block, thread)
- Global vs shared memory access
- Boundary handling
- Performance benchmarking

## Results
Shared-memory optimized kernels achieved significant speedups over
CPU and naive GPU implementations.

## Environment
- CUDA Toolkit
- NVIDIA GPU
