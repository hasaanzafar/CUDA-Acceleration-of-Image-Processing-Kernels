#include <vector>
#include <iostream>

/*
 CPU reference implementation of 2D convolution
 - Single-threaded
 - Used as correctness and performance baseline
*/

void conv2d_cpu(
    const float* input,
    const float* kernel,
    float* output,
    int width,
    int height,
    int ksize
) {
    int half = ksize / 2;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {

            float sum = 0.0f;

            for (int ky = -half; ky <= half; ky++) {
                for (int kx = -half; kx <= half; kx++) {

                    int ix = x + kx;
                    int iy = y + ky;

                    if (ix >= 0 && ix < width &&
                        iy >= 0 && iy < height) {
