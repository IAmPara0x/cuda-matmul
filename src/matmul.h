#ifndef MATMUL_H

#define MATMUL_H 1

#include <cstddef>
#include <cublas_v2.h>
#include <cuda_runtime.h>

constexpr int THREADS = 32;

__global__ void MatMulKernel_Naive(float *A, float *B, float *C,
                                   size_t N);

__global__ void MatMulKernel_FMA(float *A, float *B, float *C, size_t N);

__global__ void MatMulKernel_Strided(float *A, float *B, float *C,
                                     size_t N);
__global__ void MatMulKernel_RowMajor(float *A, float *B, float *C,
                                      size_t N);
void cuBlas_MatMul(cublasHandle_t handle, float *dA, float *dB, float *dC,
                   size_t N);

#endif

