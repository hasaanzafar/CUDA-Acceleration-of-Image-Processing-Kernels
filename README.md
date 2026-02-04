CUDA Acceleration of Image Processing Kernels
Overview

This project explores GPU concurrency and memory optimization by implementing custom CUDA kernels for image processing operations, focusing on 2D convolution. The goal is to study how parallelism, memory hierarchy, and kernel configuration impact performance.

The work is inspired by concepts from Concurrent Programming with GPUs (Coursera) and emphasizes performance analysis rather than raw accuracy.

Implemented Kernels

Naive CUDA Convolution

One thread per output pixel

Direct global memory access

Shared Memory Optimized Convolution

Tiled convolution using shared memory

Reduced global memory accesses

Explicit thread synchronization (__syncthreads())

Benchmarking & Evaluation

CUDA kernels were benchmarked against:

CPU convolution (single-threaded)

PyTorch CPU implementation

PyTorch CUDA implementation (reference baseline)

Metrics analyzed:

Execution latency

Throughput

Memory-access behavior

Block-size sensitivity

Block Size Sensitivity Study

To analyze concurrency and occupancy, convolution kernels were evaluated using different block sizes:

8×8

16×16

32×32

Results demonstrate the impact of thread-block configuration on execution efficiency.

(See results/block_size_comparison.png)

Key Learnings

Memory access patterns significantly affect performance.

Shared memory tiling reduces global memory latency.

Kernel configuration (block size) influences occupancy and throughput.

Highly optimized libraries (e.g., PyTorch) outperform custom kernels but provide useful reference points.

Technologies Used

CUDA C++

Python (benchmarking & visualization)

PyTorch (baseline comparison)

CMake
