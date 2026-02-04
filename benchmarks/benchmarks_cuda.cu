#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "../src/cuda/utils.cuh"

/* Kernel launchers */
void launch_conv2d_naive(
    const float* d_input,
    const float* d_kernel,
    float* d_output,
    int width,
    int height,
    int ksize
);

void launch_conv2d_shared(
    const float* d_input,
    const float* d_kernel,
    float* d_output,
    int width,
    int height,
    int ksize
);

int main() {
    const int width = 512;
    const int height = 512;
    const int ksize = 3;

    const int img_size = width * height;
    const int kernel_size = ksize * ksize;

    std::vector<float> h_input(img_size, 1.0f);
    std::vector<float> h_kernel(kernel_size, 1.0f / kernel_size);

    float *d_input, *d_kernel, *d_output;

    CUDA_CHECK(cudaMalloc(&d_input, img_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_kernel, kernel_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, img_size * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(
        d_input, h_input.data(),
        img_size * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        d_kernel, h_kernel.data(),
        kernel_size * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    /* -------- Naive -------- */
    GPUTimer timer;
    timer.tic();
    launch_conv2d_naive(
        d_input, d_kernel, d_output,
        width, height, ksize
    );
    float naive_ms = timer.toc();

    /* -------- Shared -------- */
    timer.tic();
    launch_conv2d_shared(
        d_input, d_kernel, d_output,
        width, height, ksize
    );
    float shared_ms = timer.toc();

    std::cout << "Naive CUDA time: "
              << naive_ms << " ms" << std::endl;
    std::cout << "Shared CUDA time: "
              << shared_ms << " ms" << std::endl;

    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);

    return 0;
}
