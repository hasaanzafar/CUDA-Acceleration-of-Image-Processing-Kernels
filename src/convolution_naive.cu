#include <cuda_runtime.h>
#include <stdio.h>

#define KERNEL_SIZE 3
#define RADIUS (KERNEL_SIZE / 2)

// Naive 2D convolution
__global__ void conv2d_naive(
    const float* input,
    const float* kernel,
    float* output,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float value = 0.0f;

    for (int ky = -RADIUS; ky <= RADIUS; ky++) {
        for (int kx = -RADIUS; kx <= RADIUS; kx++) {
            int ix = x + kx;
            int iy = y + ky;

            if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                float pixel = input[iy * width + ix];
                float weight = kernel[(ky + RADIUS) * KERNEL_SIZE + (kx + RADIUS)];
                value += pixel * weight;
            }
        }
    }

    output[y * width + x] = value;
}

void launch_conv2d_naive(
    float* d_input,
    float* d_kernel,
    float* d_output,
    int width,
    int height
) {
    dim3 block(16, 16);
    dim3 grid(
        (width + block.x - 1) / block.x,
        (height + block.y - 1) / block.y
    );

    conv2d_naive<<<grid, block>>>(
        d_input, d_kernel, d_output, width, height
    );
}
