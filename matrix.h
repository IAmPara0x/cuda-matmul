#ifndef MATRIX_H

#define MATRIX_H

#include <iostream>
#include <functional>

template <typename T>
inline size_t matrix_size(T* mat, size_t N) {
    return N * N * sizeof(T);
}

void print_mat(float *matrix, size_t N);
float* random_mat(size_t N, int hi);
void transpose(float *matrix, size_t N);
float* init_mat(size_t N);
void sgemm_cpu(float *A, float *B, float *C, size_t N);
bool same_matrix(float *A, float *B, size_t N, float tolerance);

float measure_gflops(std::function<void()> MatMul_kernel, size_t N, size_t iterations);

#endif
