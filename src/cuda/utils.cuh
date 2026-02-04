#pragma once
#include <cuda_runtime.h>
#include <iostream>

/* ---------------- CUDA Error Checking ---------------- */
#define CUDA_CHECK(call)                                       \
    do {                                                       \
        cudaError_t err = call;                                \
        if (err != cudaSuccess) {                              \
            std::cerr << "CUDA error at " << __FILE__          \
                      << ":" << __LINE__ << " — "              \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE);                                \
        }                                                      \
    } while (0)

/* ---------------- GPU Timer ---------------- */
struct GPUTimer {
    cudaEvent_t start, stop;

    GPUTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GPUTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void tic() {
        cudaEventRecord(start);
    }

    float toc() {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }
};
