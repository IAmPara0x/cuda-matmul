#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include "matrix.h"
#include "matmul.h"

using namespace std;

#define THREADS 32

int main(void) {

    getDeviceInfo();

    Matrix A = readMat("A.txt");
    Matrix B = readMat("B.txt");
    Matrix C = readMat("C.txt");
    Matrix result = alloc_mat(A.rows, A.cols);

    cout << "Matrix Size: " << A.rows << "x" << A.cols << endl;

    float *cuMat1 = nullptr, *cuMat2 = nullptr, *cuMatResult = nullptr;

    cudaMalloc((void **)&cuMat1, A.size);
    cudaMalloc((void **)&cuMat2, B.size);
    cudaMalloc((void **)&cuMatResult, A.size);

    // cuMemcpy()
    cudaMemcpy(cuMat1, A.value, A.size, cudaMemcpyHostToDevice);
    cudaMemcpy(cuMat2, B.value, A.size, cudaMemcpyHostToDevice);


    // NOTE:: START TIMER

    auto start = std::chrono::high_resolution_clock::now();
    
    dim3 threadsPerBlock(THREADS, THREADS);
    dim3 blocks((A.rows + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (A.rows + threadsPerBlock.y - 1) / threadsPerBlock.y);
    MatMulKernel<<<blocks, threadsPerBlock>>>(cuMat1, cuMat2, cuMatResult, A.rows);
    cudaDeviceSynchronize();

    // NOTE:: END TIMER
    auto end = std::chrono::high_resolution_clock::now();

    cudaMemcpy(result.value, cuMatResult, A.size, cudaMemcpyDeviceToHost);

    // cudaFree
    cudaFree(cuMat1);
    cudaFree(cuMat2);
    cudaFree(cuMatResult);

    std::chrono::duration<double> duration = end - start;

    if (result == C) 
        cout << "Test Passed!" << endl;
    else
        cout << "Test Failed!" << endl;

    std::cout << "Execution time: " << duration.count() << " seconds" << std::endl;

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
