#include "matmul.h"
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void MatMulKernel_Naive(float *A, float *B, float *C,
                                   size_t N) {

  int row = threadIdx.y + blockIdx.y * blockDim.y;
  int col = threadIdx.x + blockIdx.x * blockDim.x;

  float tmp = 0;

  for (int i = 0; i < N; i++)
    tmp += A[row * N + i] * B[i * N + col];

  C[row * N + col] = tmp;
}

__global__ void MatMulKernel_RowMajor(float *A, float *B, float *C, size_t N) {

  int row = threadIdx.y + blockIdx.y * blockDim.y;
  int col = threadIdx.x + blockIdx.x * blockDim.x;

  float tmp = 0;

  int offsetR = row * N;
  int offsetC = col * N;

  for (int i = 0; i < N; i++)
    tmp += A[offsetR + i] * B[offsetC + i];

  C[row * N + col] = tmp;
}

__global__ void MatMulKernel_FMA(float *A, float* B, float *C, size_t N)
{

  int row = fmaf(blockIdx.y, blockDim.y, threadIdx.y);
  int col = fmaf(blockIdx.x, blockDim.x, threadIdx.x);

  float sum = 0;

  int offsetR = row * N;
  int offsetC = col * N;

  for (int i = 0; i < N; i++)
    sum += A[offsetR + i] * B[offsetC + i];

  C[row * N + col] = sum;
}

// Matrix Multiplication Kernel for square matrix
__global__ void MatMulKernel_Strided(float *A, float *B, float *C,
                                     size_t N) {

  __shared__ float sA[THREADS][THREADS];
  __shared__ float sB[THREADS][THREADS];

  int row = fmaf(blockIdx.y, blockDim.y, threadIdx.y);
  int col = fmaf(blockIdx.x, blockDim.x, threadIdx.x);

  float sum = 0.0;

  int GRID = blockDim.y;
  int COL = 0;
  int X1 = fmaf(blockIdx.y, GRID, threadIdx.y) * N;
  int X2 = fmaf(blockIdx.x, GRID, threadIdx.y) * N;

  for (int i = 0; i < N; i += GRID) {

    COL = threadIdx.x + i;

    sA[threadIdx.y][threadIdx.x] = A[X1 + COL];
    sB[threadIdx.y][threadIdx.x] = B[X2 + COL];

    __syncthreads();

    for (int j = 0; j < THREADS; j++)
      sum += sA[threadIdx.y][j] * sB[threadIdx.x][j];

    __syncthreads();
  };

  C[row * N + col] = sum;
}


void cuBlas_MatMul(cublasHandle_t handle, float *dA, float *dB, float *dC,
                   size_t N) {

  const float alpha = 1.0f;
  const float beta = 0.5f;

  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, dA, N, dB, N,
              &beta, dC, N);
}


