#include "matmul.h"
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void MatMulKernel_Naive(float *A, float *B, float *C,
                                   size_t N) {

  int row = threadIdx.y + blockIdx.y * blockDim.y;
  int col = threadIdx.x + blockIdx.x * blockDim.x;

  float tmp = 0;

  for (int i = 0; i < N; i++)
    tmp = fmaf(A[row * N + i], B[i * N + col], tmp);

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

// Matrix Multiplication Kernel for square matrix
__global__ void MatMulKernel_Strided(float *A, float *B, float *C,
                                     size_t N) {

  __shared__ float sA[THREADS][THREADS];
  __shared__ float sB[THREADS][THREADS];

  int row = fmaf(blockIdx.y, blockDim.y, threadIdx.y);
  int col = fmaf(blockIdx.x, blockDim.x, threadIdx.x);

  float sum = 0.0;

  int GRID = blockDim.y;

  int x1,y2;

  x1 = (blockIdx.y * THREADS + threadIdx.y) * N;
  y2 = (THREADS * blockIdx.x + threadIdx.x);


  for (int i = 0; i < N; i += GRID) {

    sA[threadIdx.y][threadIdx.x] = A[x1 + threadIdx.x + i];
    sB[threadIdx.y][threadIdx.x] = B[(threadIdx.y + i) * N + y2];


    __syncthreads();

    for (int j = 0; j < THREADS; j++)
      sum += sA[threadIdx.y][j] * sB[j][threadIdx.x];

    __syncthreads();

  }

  C[row * N + col] = sum;
}


void cuBlas_MatMul(cublasHandle_t handle, float *dA, float *dB, float *dC,
                   size_t N) {

  const float alpha = 1.0f;
  const float beta = 0.5f;

  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, dA, N, dB, N,
              &beta, dC, N);
}


