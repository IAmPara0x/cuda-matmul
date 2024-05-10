#ifndef MATMUL_H

#define MATMUL_H 1

#include <cstddef>
#include <cuda_runtime.h>
#include <cublas_v2.h>

constexpr int THREADS = 16;

__global__ void MatMulKernelNaive(float *mat1, float *mat2, float *result, size_t N);
__global__ void MatMulKernelStrided(float *mat1, float *mat2, float *result, size_t N);
__global__ void MatMulKernelRowMajor(float *mat1, float *mat2, float *result, size_t N);
void cuBlas_MatMul(cublasHandle_t handle, float *dA, float *dB, float *dC, size_t N);

#endif
