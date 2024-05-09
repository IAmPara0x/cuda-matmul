#ifndef MATMUL_H

#define MATMUL_H 1

#include <cstddef>
#include <cuda_runtime.h>

constexpr int THREADS = 16;

__global__ void MatMulKernel(float *mat1, float *mat2, float *result, size_t N);

#endif
