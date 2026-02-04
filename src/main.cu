#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>

/* CPU reference */
void conv2d_cpu(
    const float* input,
    const float* kernel,
    float* output,
    int width,
    int height,
    int ksize
);

/* CUDA kernels */
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

/* Utility */
void check_cuda(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: "
                  << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    const int width = 512;
    const int height = 512;
    const int ksize = 3;

    const int img_size = width * height;
    const int kernel_size = ksize * ksize;

    std::vector<float> h_input(img_size, 1.0f);
    std::vector<float> h_kernel(kernel_size, 1.0f / kernel_size);

    std::vector<float> h_output_cpu(img_size);
    std::vector<float> h_output_gpu(img_size);

    /* ---------------- CPU ---------------- */
    auto cpu_start = std::chrono::high_resolution_clock::now();
    conv2d_cpu(
        h_input.data(),
        h_kernel.data(),
        h_output_cpu.data(),
        width,
        height,
        ksize
    );
    auto cpu_end = std::chrono::high_resolution_clock::now();

    double cpu_time =
        std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    std::cout << "CPU time: " << cpu_time << " ms" << std::endl;

    /* ---------------- GPU memory ---------------- */
    float *d_input, *d_kernel, *d_output;

    check_cuda(cudaMalloc(&d_input, img_size * sizeof(float)));
    check_cuda(cudaMalloc(&d_kernel, kernel_size * sizeof(float)));
    check_cuda(cudaMalloc(&d_output, img_size * sizeof(float)));

    check_cuda(cudaMemcpy(
        d_input, h_input.data(),
        img_size * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    check_cuda(cudaMemcpy(
        d_kernel, h_kernel.data(),
        kernel_size * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    /* ---------------- Naive CUDA ---------------- */
    auto gpu_start = std::chrono::high_resolution_clock::now();
    launch_conv2d_naive(
        d_input, d_kernel, d_output,
        width, height, ksize
    );
    auto gpu_end = std::chrono::high_resolution_clock::now();

    double naive_time =
        std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count();

    std::cout << "Naive CUDA time: "
              << naive_time << " ms" << std::endl;

    /* ---------------- Shared CUDA ---------------- */
    gpu_start = std::chrono::high_resolution_clock::now();
    launch_conv2d_shared(
        d_input, d_kernel, d_output,
        width, height, ksize
    );
    gpu_end = std::chrono::high_resolution_clock::now();

    double shared_time =
        std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count();

    std::cout << "Shared CUDA time: "
              << shared_time << " ms" << std::endl;

    /* ---------------- Cleanup ---------------- */
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);

    return 0;
}
