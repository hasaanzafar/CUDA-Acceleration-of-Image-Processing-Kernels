#include <cuda_runtime.h>
#include <stdio.h>

/*
 Naive 2D convolution kernel
 - One thread computes one output pixel
 - Direct global memory access
 - No shared memory optimization
*/

__global__
void conv2d_naive(
    const float* input,
    const float* kernel,
    float* output,
    int width,
    int height,
    int ksize
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int half = ksize / 2;
    float sum = 0.0f;

    for (int ky = -half; ky <= half; ky++) {
        for (int kx = -half; kx <= half; kx++) {
            int ix = x + kx;
            int iy = y + ky;

            if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                float pixel = input[iy * width + ix];
                float weight = kernel[(ky + half) * ksize + (kx + half)];
                sum += pixel * weight;
            }
        }
    }

    output[y * width + x] = sum;
}

/*
 Host launcher
*/
void launch_conv2d_naive(
    const float* d_input,
    const float* d_kernel,
    float* d_output,
    int width,
    int height,
    int ksize
) {
    dim3 block(16, 16);
    dim3 grid(
        (width + block.x - 1) / block.x,
        (height + block.y - 1) / block.y
    );

    conv2d_naive<<<grid, block>>>(
        d_input,
        d_kernel,
        d_output,
        width,
        height,
        ksize
    );

    cudaDeviceSynchronize();
}
