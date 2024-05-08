#include <cuda_runtime.h>
#include <stdio.h>


// Matrix Multiplication Kernel for square matrix
__global__ void MatMulKernel(float *mat1, float *mat2, float *result, size_t N) {

    // __shared__ float A[32 * 32];
    // __shared__ float B[32 * 32];
    //
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    // A[idx] = mat1[idx];
    // B[idx] = mat2[idx];
    //
    // __syncthreads();

    float tmp = 0;

    for (int i = 0; i < N; i++)
        tmp += mat1[row*N + i] * mat2[i*N + col];

    result[row * N + col] = tmp;
}
