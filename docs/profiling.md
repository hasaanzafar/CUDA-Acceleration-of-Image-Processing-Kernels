# CUDA Convolution Profiling and Performance Analysis

## 1. Objective
This document analyzes the performance characteristics of custom CUDA convolution kernels,
with emphasis on GPU concurrency, memory hierarchy utilization, and kernel configuration.

The study is motivated by concepts from *Concurrent Programming with GPUs (Coursera)* and
focuses on understanding performance trade-offs rather than outperforming optimized libraries.

---

## 2. Experimental Setup

### Hardware
- GPU: NVIDIA T4 (Kaggle) / RTX-class (local)
- CPU: x86_64
- CUDA Version: 12.x

### Input
- Image: 512×512 grayscale PNG
- Kernel: 3×3 convolution filter
- Data type: float32

### Baselines
- Single-threaded CPU convolution
- PyTorch CPU implementation
- PyTorch CUDA implementation

---

## 3. Kernels Evaluated

### 3.1 Naive CUDA Convolution
- One thread per output pixel
- Direct global memory access
- No shared memory usage

This kernel maximizes parallelism but suffers from redundant global memory loads.

---

### 3.2 Shared Memory Optimized Convolution
- Tiled convolution using shared memory
- Halo regions loaded cooperatively
- Explicit synchronization using `__syncthreads()`

This approach reduces global memory traffic at the cost of increased shared memory usage.

---

## 4. Block Size Sensitivity Study

Kernels were evaluated using the following block sizes:
- 8×8
- 16×16
- 32×32

### Observations
- 8×8 blocks underutilize available SM resources
- 16×16 achieves balanced occupancy and memory reuse
- 32×32 increases register pressure and limits occupancy on some architectures

Block configuration had a measurable impact on throughput and latency.

---

## 5. Profiling Results

### Memory Behavior
- Naive kernel shows high global load transactions per output pixel
- Shared memory kernel significantly reduces global memory accesses
- Memory coalescing improves with tiled access patterns

### Execution Characteristics
- Shared memory kernel exhibits lower execution latency
- Synchronization overhead is amortized over reduced memory traffic
- Occupancy is bounded by shared memory usage per block

---

## 6. Comparison with PyTorch

PyTorch CUDA implementations outperform custom kernels due to:
- Advanced kernel fusion
- Architecture-specific optimizations
- Highly tuned memory access strategies

However, custom kernels demonstrate predictable performance trends and validate theoretical expectations.

---

## 7. Key Takeaways

- Memory access patterns dominate performance more than arithmetic complexity
- Shared memory is effective when reuse outweighs synchronization cost
- Kernel launch configuration strongly influences occupancy and throughput
- Highly optimized libraries serve as valuable performance references

---

## 8. Conclusion

This project demonstrates how CUDA kernel design decisions impact concurrency,
memory efficiency, and overall performance. While custom kernels do not surpass
library implementations, they provide critical insight into GPU execution behavior
and optimization strategies.
