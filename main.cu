#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <cublas_v2.h>
#include "matrix.h"
#include "matmul.h"

using namespace std;


void square_matmul(float *A, float *B, float *C, size_t N) {

    // NOTE:: START TIMER
    auto start = std::chrono::high_resolution_clock::now();

    dim3 threadsPerBlock(THREADS, THREADS);
    dim3 blocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
    MatMulKernel<<<blocks, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();

    // NOTE:: END TIMER
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;
    std::cout << "Execution time: " << duration.count() << " seconds" << std::endl;
}

void cublas_matmul(cublasHandle_t handle, float *A, float *B, float *C, size_t N) {

    // Initialize cuBLAS context

    // Parameters for matrix multiplication
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // NOTE:: START TIMER

    auto start = std::chrono::high_resolution_clock::now();
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, A, N, B, N, &beta, C, N);

    // NOTE:: END TIMER
    auto end = std::chrono::high_resolution_clock::now();

}


int main(void) {

    getDeviceInfo();

    cublasHandle_t handle;
    cublasCreate(&handle);

    Matrix A = readMat("A.txt");
    Matrix B = readMat("B.txt");
    Matrix C = readMat("C.txt");
    Matrix result = alloc_mat(A.rows, A.cols);

    transpose(&B);
    
    cout << "Matrix Size: " << A.rows << "x" << A.cols << endl;

    float *cuMat1 = nullptr, *cuMat2 = nullptr, *cuMatResult = nullptr;

    cudaMalloc((void **)&cuMat1, A.size);
    cudaMalloc((void **)&cuMat2, B.size);
    cudaMalloc((void **)&cuMatResult, A.size);

    // cuMemcpy()
    cudaMemcpy(cuMat1, A.value, A.size, cudaMemcpyHostToDevice);
    cudaMemcpy(cuMat2, B.value, A.size, cudaMemcpyHostToDevice);

    square_matmul(cuMat1, cuMat2, cuMatResult, A.rows);
    // cublas_matmul(handle, cuMat1, cuMat2, cuMatResult, A.rows);

    cudaMemcpy(result.value, cuMatResult, A.size, cudaMemcpyDeviceToHost);

    // cudaFree
    cudaFree(cuMat1);
    cudaFree(cuMat2);
    cudaFree(cuMatResult);
    cublasDestroy(handle);

    if (result == C) 
        cout << "Test Passed!" << endl;
    else
        cout << "Test Failed!" << endl;


    // host free
    free_mat(A);
    free_mat(B);
    free_mat(C);
    free_mat(result);
    return 0;
}


void getDeviceInfo() {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    
    std::cout << "Device Name: " << props.name << std::endl;
    std::cout << "Number of SMs: " << props.multiProcessorCount << std::endl;
    std::cout << "Max Threads per SM: " << props.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "Max Threads per Block: " << props.maxThreadsPerBlock << std::endl;
    std::cout << "Shared Memory per Block: " << props.sharedMemPerBlock << std::endl;
    return;
};
