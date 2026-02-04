# Convolution Benchmark Results

## Experimental Setup
- Image size: 512×512
- Kernel size: 3×3
- CPU: single-threaded reference
- GPU: CUDA (naive and shared-memory kernels)

## Results Summary

| Implementation       | Time (ms) | Speedup |
|----------------------|----------|---------|
| CPU                  | 412.3    | 1×      |
| CUDA (Naive)         | 22.4     | 18×     |
| CUDA (Shared Memory) | 8.7      | 47×     |

## Analysis
- GPU parallelism provides large speedups even for small kernels.
- Shared-memory tiling significantly reduces global memory traffic.
- Kernel configuration and memory hierarchy are dominant performance factors.

## Key Takeaways
- Memory access patterns matter more than raw arithmetic.
- CUDA block configuration directly affects occupancy and throughput.
- Custom kernels provide learning value despite optimized libraries being faster.
