# CUDA Acceleration of Image Processing Kernels

## Overview
This project explores GPU concurrency and memory optimization by implementing custom CUDA kernels for image processing operations, focusing on 2D convolution. The goal is to study how parallelism, memory hierarchy, and kernel configuration impact performance.

The work is inspired by concepts from Concurrent Programming with GPUs (Coursera) and emphasizes performance analysis rather than raw accuracy.

Implemented Kernels
1. Naive CUDA Convolution
2. One thread per output pixel
3. Direct global memory access

Shared Memory Optimized Convolution
1. Tiled convolution using shared memory
2. Reduced global memory accesses
3. Explicit thread synchronization (__syncthreads())

## Benchmarking & Evaluation

CUDA kernels were benchmarked against:
1. CPU convolution (single-threaded)
2. PyTorch CPU implementation
3. PyTorch CUDA implementation (reference baseline)

Metrics analyzed:
1. Execution latency
2. Throughput
3. Memory-access behavior
4. Block-size sensitivity

## Block Size Sensitivity Study

To analyze concurrency and occupancy, convolution kernels were evaluated using different block sizes:
1. 8×8
2. 16×16
3. 32×32

Results demonstrate the impact of thread-block configuration on execution efficiency.

(See results/block_size_comparison.png)

## Key Learnings
1. Memory access patterns significantly affect performance.
2. Shared memory tiling reduces global memory latency.
3. Kernel configuration (block size) influences occupancy and throughput.
4. Highly optimized libraries (e.g., PyTorch) outperform custom kernels but provide useful reference points.

## Technologies Used
1. CUDA C++
2. Python (benchmarking & visualization)
3. PyTorch (baseline comparison)

CMake
