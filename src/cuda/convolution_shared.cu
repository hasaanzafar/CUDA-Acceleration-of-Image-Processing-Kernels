#include <cuda_runtime.h>
#include <stdio.h>

/*
 Shared-memory optimized 2D convolution
 - Tiled convolution
 - Shared memory for input tile
 - Explicit synchronization
*/

#define TILE_SIZE 16
#define MAX_KERNEL_SIZE 7   // supports up to 7x7 kernels

__global__
void conv2d_shared(
    const float* input,
    const float* kernel,
    float* output,
    int width,
    int height,
    int ksize
) {
    __shared__ float tile[TILE_SIZE + MAX_KERNEL_SIZE - 1]
                          [TILE_SIZE + MAX_KERNEL_SIZE - 1];

    int half = ksize / 2;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int x = blockIdx.x * TILE_SIZE + tx;
    int y = blockIdx.y * TILE_SIZE + ty;

    int shared_x = tx + half;
    int shared_y = ty + half;

    /* Load main tile pixel */
    if (x < width && y < height)
        tile[shared_y][shared_x] = input[y * width + x];
    else
        tile[shared_y][shared_x] = 0.0f;

    /* Load halo regions */
    if (tx < half) {
        int lx = x - half;
        tile[shared_y][tx] =
            (lx >= 0 && y < height) ? input[y * width + lx] : 0.0f;

        int rx = x + TILE_SIZE;
        tile[shared_y][shared_x + TILE_SIZE] =
            (rx < width && y < height) ? input[y * width + rx] : 0.0f;
    }

    if (ty < half) {
        int uy = y - half;
        tile[ty][s]()
