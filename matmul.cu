#include <cuda_runtime.h>
#include <stdio.h>
#include "matmul.h"


// Matrix Multiplication Kernel for square matrix
__global__ void MatMulKernel(float *mat1, float *mat2, float *result, size_t N) {

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

