#ifndef _MATMUL_KERNEL_

#define _MATMUL_KERNEL_

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>


#define THREADS 32

__global__ void MatMulKernel_Naive(float *A, float *B, float *C,
                                   size_t N) {

  uint row = threadIdx.y + blockIdx.y * blockDim.y;
  uint col = threadIdx.x + blockIdx.x * blockDim.x;

  float tmp = 0;

  for (uint i = 0; i < N; i++)
    tmp = fmaf(A[row * N + i], B[i * N + col], tmp);

  C[row * N + col] = tmp;

}


template<const uint BLOCKSIZE>
__global__ void MatMulKernel_GlobalCoalesce(float *A, float *B, float *C,
                                   size_t N) {

  uint row = blockIdx.y * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  uint col = blockIdx.x * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

  float tmp = 0;

  for (uint i = 0; i < N; i++)
    tmp = fmaf(A[row * N + i], B[i * N + col], tmp);

  C[row * N + col] = tmp;

}


template<const uint BLOCKSIZE>
__global__ void MatMulKernel_Strided(float *A, float *B, float *C,
                                     size_t N) {

  __shared__ float sA[BLOCKSIZE][BLOCKSIZE];
  __shared__ float sB[BLOCKSIZE][BLOCKSIZE];


  uint threadRow = threadIdx.x / BLOCKSIZE;
  uint threadCol = threadIdx.x % BLOCKSIZE;

  float sum = 0.0;

  uint x1,y2;

  x1 = (blockIdx.y * BLOCKSIZE + threadRow) * N;
  y2 = (BLOCKSIZE * blockIdx.x + threadCol);


  for (uint i = 0; i < N; i += BLOCKSIZE) {

    sA[threadRow][threadCol] = A[x1 + threadCol + i];
    sB[threadRow][threadCol] = B[(uint)fmaf((threadRow + i), N, y2)];


    __syncthreads();

    for (uint j = 0; j < BLOCKSIZE; j++)
      sum = fmaf(sA[threadRow][j], sB[j][threadCol], sum);

    __syncthreads();

  }

  C[(blockIdx.y * BLOCKSIZE + threadRow) * N + (blockIdx.x * BLOCKSIZE + threadCol)] = sum;
}


template<const uint H, const uint  W, const uint K>
__global__ void MatMulKernel_1DBlockTiling(float *A, float *B, float *C, size_t N) {



  // sA=(H,K), sB=(K,W)
  __shared__ float sA[THREADS][THREADS];
  __shared__ float sB[THREADS][THREADS];

  uint col = fmaf(blockIdx.x, blockDim.x, threadIdx.x);

  float tmpB = 0.0, results[THREADS] = {0.0f};

  uint x1,y2;

  x1 = (blockIdx.y * THREADS + threadIdx.y) * N;
  y2 = (THREADS * blockIdx.x + threadIdx.x);


  for (uint i = 0; i < N; i += blockDim.y) {

    sA[threadIdx.y][threadIdx.x] = A[x1 + threadIdx.x + i];
    sB[threadIdx.y][threadIdx.x] = B[(uint)fmaf((threadIdx.y + i), N, y2)];

    __syncthreads();

    for (uint j = 0; j < THREADS; j++)
    {
      tmpB = sB[j][threadIdx.x];
      for (uint k = 0; k < THREADS; k++)
        results[k] = fmaf(sA[k][j], tmpB, results[k]);
    }
    __syncthreads();
  }

  for (uint i = 0; i < THREADS; i++)
    C[(THREADS * blockIdx.y + i) * N + col] = results[i];
}


void cuBlas_MatMul(cublasHandle_t handle, float *dA, float *dB, float *dC,
                   size_t N) {

  const float alpha = 1.0f;
  const float beta = 0.0f;

  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, dA, N, dB, N,
              &beta, dC, N);
}


#endif
