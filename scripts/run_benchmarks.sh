#!/bin/bash
set -e

echo "Running CPU benchmark..."
./build/benchmark_cpu

echo ""
echo "Running CUDA benchmark..."
./build/benchmark_cuda
