#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <cublas_v2.h>
#include "matrix.h"
#include "matmul.h"

using namespace std;

constexpr int N=1024;
constexpr size_t iterations=1;

// #define VERIFY
// #define CUBLAS_BENCH 1

void getDeviceInfo();

int main(void) {

    getDeviceInfo();

    cublasHandle_t handle;
    cublasCreate(&handle);

    float *hA = random_mat(N, 5), *hB = random_mat(N, 5), *hC = init_mat(N), *cRef = init_mat(N);
    size_t sizeA=matrix_size(hA, N),sizeB=matrix_size(hB, N),sizeC=matrix_size(hC, N);
    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    float gflops;

    cudaMalloc((void **)&dA, sizeA);
    cudaMalloc((void **)&dB, sizeB);
    cudaMalloc((void **)&dC, sizeC);



#ifdef CUBLAS_BENCH

    cudaMemcpy(dA, hA, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sizeB, cudaMemcpyHostToDevice);

    gflops =  measure_gflops([&handle, dA, dB, dC]() { cuBlas_MatMul(handle,dA,dB,dC, N); }, N, iterations);
    printf("cuBLAS performance: (%f) GFLOPS. size: %d\n", gflops, N);

#else

    cudaMemcpy(dA, hA, sizeA, cudaMemcpyHostToDevice);
    transpose(hB, N);
    cudaMemcpy(dB, hB, sizeB, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(THREADS, THREADS);
    dim3 blocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    gflops =  measure_gflops( [&handle, dA, dB, dC, threadsPerBlock, blocks]() { MatMulKernelStrided<<<blocks, threadsPerBlock>>>(dA,dB,dC, N); }, N, iterations);

    printf("our performance: (%f) GFLOPS. size: %d\n", gflops, N);

#endif

    cudaMemcpy(hC, dC, sizeC, cudaMemcpyDeviceToHost);

#ifdef VERIFY

    transpose(hB, N);
    sgemm_cpu(hA,hB, cRef, N);

    if (same_matrix(hC,cRef, N, 1e-1))
        cout << "Verified!" << endl;
    else
        cout << "Verification Failed!" << endl;
#endif

    // cuda free, matrices:
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cublasDestroy(handle);

    // free matrices: 
    free(hA);
    free(hB);
    free(hC);
    free(cRef);
};


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
