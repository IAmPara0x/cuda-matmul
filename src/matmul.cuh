#ifndef _MATMUL_KERNEL_

#define _MATMUL_KERNEL_

#include <cassert>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>


__global__ void MatMulKernel_Naive(float *A, float *B, float *C,
                                   size_t N) {

  uint col = threadIdx.y + blockIdx.y * blockDim.y;
  uint row = threadIdx.x + blockIdx.x * blockDim.x;

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

  A += (blockIdx.y * BLOCKSIZE + threadRow) * N;
  B += (BLOCKSIZE * blockIdx.x + threadCol);

  for (uint i = 0; i < N; i += BLOCKSIZE) {

    sA[threadRow][threadCol] = A[threadCol];
    sB[threadRow][threadCol] = B[threadRow * N];

    __syncthreads();

    for (uint j = 0; j < BLOCKSIZE; j++)
      sum = fmaf(sA[threadRow][j], sB[j][threadCol], sum);

    __syncthreads();

    A += BLOCKSIZE;
    B += BLOCKSIZE * N;

  }

  C[(blockIdx.y * BLOCKSIZE + threadRow) * N + (blockIdx.x * BLOCKSIZE + threadCol)] = sum;
}

template<const uint DM, const uint DK, const uint T>
__global__ void MatMulKernel_1DBlockTiling(float *A, float *B, float *C,
                                     size_t N) {

  // TODO: FIND a general way to reduce bank conflict
  __shared__ float sA[DK][DM + 4];
  __shared__ float sB[DK][DM];


  float tmpB = 0.0f, result[T] = {0.0f};

  uint colA, rowA, colB, rowB;

  colA = threadIdx.x % DK;
  rowA = threadIdx.x / DK;

  colB = threadIdx.x % DM;
  rowB = threadIdx.x / DM;


  A += (blockIdx.y * DM + rowA) * N;
  B += DM * blockIdx.x;


  for (uint i = 0; i < N; i += DK) {

    sA[colA][rowA] = A[colA];
    sB[rowB][colB] = B[rowB * N + colB];

    __syncthreads();

    for (uint j = 0; j < DK; j++)
    {
      tmpB = sB[j][colB];

      for (uint k = 0; k < T; k++)
        result[k] += sA[j][(rowB * T) + k] * tmpB;
    }

    __syncthreads();

    A += DK;
    B += DK * N;

  }

  C += blockIdx.y * DM * N;

  uint col = fmaf(blockIdx.x, DM, colB);
  for (uint i = 0; i < T; i++)
    C[(rowB * T + i) * N + col] = result[i];

}


template<const uint DM, const uint DK, const uint TM, const uint TK>
__global__ void MatMulKernel_2DBlockTiling(float *A, float *B, float *C,
                                     size_t N) {


  assert (TM % 4 == 0 && "TM must be divisble by 4.");
  assert (TK % 4 == 0 && "TK must be divisble by 4.");

  // TODO: Find a general way to add padding
  __shared__ float sA[DK][DM + 4];
  __shared__ float sB[DK][DM];


  float result[TM][TK] = {0.0f}, colN[TK] = {0.0f}, rowN[TM] = {0.0f};

  uint colA, rowA, colB, rowB;

  colA = threadIdx.x % DK;
  rowA = threadIdx.x / DK;

  colB = threadIdx.x % DM;
  rowB = threadIdx.x / DM;

  const int threadCol = threadIdx.x % (DM / TK);
  const int threadRow = threadIdx.x / (DM / TK);

  A += (blockIdx.y * DM) * N;
  B += DM * blockIdx.x;


  for (uint i = 0; i < N; i += DK) {


    for (uint j = 0; j < TK; j++)
      sA[colA][rowA * TK + j] = A[(rowA * TK + j) * N + colA];

    for (uint j = 0; j < TK; j++)
      sB[rowB * TK + j][colB] = B[(rowB * TK + j) * N + colB];

    __syncthreads();

    for (uint dotIdx = 0; dotIdx < DK; dotIdx++)
    {
      for (uint colT = 0; colT < TK; colT++)
        colN[colT] = sB[dotIdx][threadCol * TK + colT];

      for (uint rowT = 0; rowT < TM; rowT++)
        rowN[rowT] = sA[dotIdx][TM * threadRow + rowT];

      for (uint colT = 0; colT < TK; colT++)
        for (uint rowT = 0; rowT < TM; rowT++)
          result[rowT][colT] += rowN[rowT] * colN[colT];
    }

    __syncthreads();

    A += DK;
    B += DK * N;

  }

  C += blockIdx.y * DM * N + blockIdx.x * DM;

  for (uint colT = 0; colT < TK; colT++)
  {
    uint col = threadCol * TK + colT;
    for (uint rowT = 0; rowT < TM; rowT++)
      C[(threadRow * TM + rowT) * N + col] = result[rowT][colT];
  }

}



template<const uint DM, const uint DK, const uint TM, const uint TK>
__global__ void MatMulKernel_Final(float *A, float *B, float *C,
                                     uint N) {

  assert (TM % 4 == 0 && "TM must be divisble by 4.");
  assert (TK % 4 == 0 && "TK must be divisble by 4.");

  // TODO: Find a general way to add padding
  __shared__ float sA[DK][DM + 4];
  __shared__ float sB[DK][DM];


  float result[TM][TK] = {0.0f}, colN[TK] = {0.0f}, rowN[TM] = {0.0f};

  uint colA, rowA, colB, rowB;

  colA = threadIdx.x % DK;
  rowA = threadIdx.x / DK;

  colB = threadIdx.x % (DM / TK);
  rowB = threadIdx.x / (DM / TK);

  const int threadCol = threadIdx.x % (DM / TK);
  const int threadRow = threadIdx.x / (DM / TK);

  A += (blockIdx.y * DM) * N;
  B += rowB * N + DM * blockIdx.x;


  for (uint i = 0; i < N; i += DK) {

    for (uint j = 0; j < TK; j += 1)
      sA[colA][rowA * TK + j] = A[(rowA * TK + j) * N + colA];

    for (uint j = 0; j < TK; j += 4)
      reinterpret_cast<float4 *>(&sB[rowB][colB * TK + j])[0] = reinterpret_cast<float4 *>(&B[colB * TK + j])[0];

    __syncthreads();

    for (uint dotIdx = 0; dotIdx < DK; dotIdx++)
    {

      for (uint colT = 0; colT < TK; colT += 4)
        reinterpret_cast<float4 *>(&colN[colT])[0] = reinterpret_cast<float4 *>(&sB[dotIdx][threadCol * TK + colT])[0];

      for (uint rowT = 0; rowT < TM; rowT += 4)
        reinterpret_cast<float4 *>(&rowN[rowT])[0] = reinterpret_cast<float4 *>(&sA[dotIdx][threadRow * TM + rowT])[0];

      for (uint colT = 0; colT < TK; colT++)
        for (uint rowT = 0; rowT < TM; rowT++)
          result[rowT][colT] += rowN[rowT] * colN[colT];
    }

    __syncthreads();

    A += DK;
    B += DK * N;

  }

  C += blockIdx.y * DM * N + blockIdx.x * DM;

  for (uint colT = 0; colT < TK; colT++)
  {
    uint col = threadCol * TK + colT;
    for (uint rowT = 0; rowT < TM; rowT++)
      C[(threadRow * TM + rowT) * N + col] = result[rowT][colT];
  }

}


void cuBlas_MatMul(cublasHandle_t handle, float *dA, float *dB, float *dC,
                   size_t N) {

  const float alpha = 1.0f;
  const float beta = 0.0f;

  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, dA, N, dB, N,
              &beta, dC, N);
}

#endif

