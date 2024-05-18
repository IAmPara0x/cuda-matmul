#include "matrix.h"
#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include <functional>
#include <stdio.h>

using namespace std;


void cudaCheck1(cudaError_t error, const char *file, int line) {
  if (error != cudaSuccess) {
    printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
           cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
};

void print_mat(float *matrix, size_t N) {
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < N; j++)
      printf("%8.3f ", matrix[i * N + j]);
    std::cout << "\n";
  }
}

void transpose(float *matrix, size_t N) {

  float tmp;

  for (size_t i = 0; i < N; i++)
    for (size_t j = 0; j < i; j++) {
      tmp = matrix[i * N + j];
      matrix[i * N + j] = matrix[j * N + i];
      matrix[j * N + i] = tmp;
    }
};

// TODO: allow use of negative value of high
float *random_mat(size_t N, int hi) {

  srand(time(NULL));

  assert(hi > 0 && "only positive values are allowed!");

  float *mat = init_mat(N);
  float r;

  for (size_t i = 0; i < N * N; ++i) {
    r = ((float)rand() / (RAND_MAX)) + 1;
    mat[i] = (rand() % hi) * r;
  }

  return mat;
}

float *init_mat(size_t N) {
  float *mat = (float *)malloc(sizeof(float) * N * N);

  for (size_t i = 0; i < N * N; ++i)
    mat[i] = 0;
  return mat;
}

void sgemm_cpu(float *A, float *B, float *C, size_t N) {

  for (size_t row = 0; row < N; row++)
    for (size_t col = 0; col < N; col++) {
      float sum = 0.0;
      for (size_t k = 0; k < N; k++)
        sum += A[row * N + k] * B[k * N + col];
      C[row * N + col] = sum;
    }
}

bool same_matrix(float *A, float *B, size_t N, float tolerance) {

  for (size_t i = 0; i < N * N; i++)
    if (fabs(A[i] - B[i]) > tolerance) {
      printf("Mismatch at (%zu,%zu), %f != %f\n", i / N, i % N, A[i], B[i]);
      return false;
    }
  return true;
}

float measure_gflops(std::function<void()> MatMul_kernel, size_t N, size_t iterations) {

  float elapsed_time = 0, total_elapsed_time = 0;
  cudaEvent_t beg, end;

  cudaCheck(cudaEventCreate(&beg));
  cudaCheck(cudaEventCreate(&end));

  size_t flops = 2 * N * N * N;

  for (size_t i = 0; i < iterations; ++i) {
    cudaCheck(cudaEventRecord(beg));
    MatMul_kernel();
    cudaCheck(cudaEventRecord(end));

    cudaCheck(cudaEventSynchronize(beg));
    cudaCheck(cudaEventSynchronize(end));
    cudaCheck(cudaEventElapsedTime(&elapsed_time, beg, end));
    elapsed_time /= 1000; // Convert to seconds
    total_elapsed_time += elapsed_time;
  }

  return (iterations * flops * 1e-9) / total_elapsed_time;
};
