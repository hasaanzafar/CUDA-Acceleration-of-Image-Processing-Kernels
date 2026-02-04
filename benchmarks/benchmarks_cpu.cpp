#include <iostream>
#include <vector>
#include <chrono>

/* CPU convolution implementation */
void conv2d_cpu(
    const float* input,
    const float* kernel,
    float* output,
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

    std::vector<float> input(img_size, 1.0f);
    std::vector<float> kernel(kernel_size, 1.0f / kernel_size);
    std::vector<float> output(img_size);

    auto start = std::chrono::high_resolution_clock::now();

    conv2d_cpu(
        input.data(),
        kernel.data(),
        output.data(),
        width,
        height,
        ksize
    );

    auto end = std::chrono::high_resolution_clock::now();
    double time_ms =
        std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "CPU convolution time: "
              << time_ms << " ms" << std::endl;

    return 0;
}
