#include <cuda_runtime.h>
#include <stdio.h>
#include "matmul.h"

__global__ void MatMulKernelNaive(float *mat1, float *mat2, float *result, size_t N) {

    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    float tmp = 0;

    for (int i = 0; i < N; i++)
        tmp += mat1[row*N + i] * mat2[i*N + col];

    result[row * N + col] = tmp;
}

__global__ void MatMulKernelRowMajor(float *mat1, float *mat2, float *result, size_t N) {

    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    float tmp = 0;

    for (int i = 0; i < N; i++)
        tmp += mat1[row*N + i] * mat2[col*N + i];

    result[row * N + col] = tmp;
}


// Matrix Multiplication Kernel for square matrix
__global__ void MatMulKernelStrided(float *mat1, float *mat2, float *result, size_t N) {

    __shared__ float A[THREADS][THREADS];
    __shared__ float B[THREADS][THREADS];

    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    float sum = 0.0;

    int GRID = blockDim.y;
    int COL = 0;
    int X1 = (blockIdx.y * GRID + threadIdx.y) * N;
    int X2 = (blockIdx.x * GRID + threadIdx.y) * N;

    for (int i = 0; i < N; i += GRID) {

        COL = threadIdx.x + i;

        A[threadIdx.y][threadIdx.x] = mat1[X1 + COL];
        B[threadIdx.y][threadIdx.x] = mat2[X2 + COL];

        __syncthreads();

        for (int j = 0; j < THREADS; j++)
            sum += A[threadIdx.y][j] * B[threadIdx.x][j];

        __syncthreads();
    };

    result[row * N + col] = sum;
}




void cuBlas_MatMul(cublasHandle_t handle, float *dA, float *dB, float *dC, size_t N) {

  const float alpha = 1.0f;
  const float beta = 0.5f;

  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, dA, N,
                     dB, N, &beta, dC, N);

}

