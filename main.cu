#include "./src/matrix.h"
#include "./src/runner.h"
#include "./src/matmul.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <string>

using namespace std;

constexpr int N = 1 << 10;
constexpr size_t iterations = 1;

void getDeviceInfo();
MatMulKernel getKernelName(int argc, char **argv);

int main(int argc, char **argv) {

  MatMulKernel kernel = getKernelName(argc, argv);

  getDeviceInfo();

  cublasHandle_t handle;
  cublasCreate(&handle);

  float *hA = random_mat(N, 5), *hB = random_mat(N, 5), *hC = init_mat(N),
        *cRef = init_mat(N);
  size_t sizeA = matrix_size(hA, N), sizeB = matrix_size(hB, N),
         sizeC = matrix_size(hC, N);
  float *dA = nullptr, *dB = nullptr, *dC = nullptr;
  float gflops;

  cudaMalloc((void **)&dA, sizeA);
  cudaMalloc((void **)&dB, sizeB);
  cudaMalloc((void **)&dC, sizeC);

  cudaMemcpy(dA, hA, sizeA, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, hB, sizeB, cudaMemcpyHostToDevice);

  auto func = runner(handle, kernel, {hA, hB, hC}, {dA, dB, dC}, N);

  gflops = measure_gflops(func, N, iterations); //

  printf("%s performance: (%f) GFLOPS. size: %d\n",
         matmulKernelToString(kernel).c_str(), gflops, N);

  cudaMemcpy(hC, dC, sizeC, cudaMemcpyDeviceToHost);

  if (argc == 3 && strcmp(argv[2], "-verify") == 0) {
    sgemm_cpu(hA, hB, cRef, N);

    if (same_matrix(hC, cRef, N, 1e-1))
      cout << "Verified!" << endl;
    else
      cout << "Verification Failed!" << endl;
  }

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

MatMulKernel getKernelName(int argc, char **argv) {

  if (argc == 1)
  {
    printf("USAGE: main <KERNEL_NAME> -verify\n");
    exit(-1);
  }

  string kernel_name = argv[1];

  MatMulKernel kernel;

  if (kernel_name == "CuBLAS")
    kernel = MatMulKernelCuBLAS;
  else if (kernel_name == "Naive")
    kernel = MatMulKernelNaive;
  else if (kernel_name == "FMA")
    kernel = MatMulKernelFMA;
  else if (kernel_name == "RowMajor")
    kernel = MatMulKernelRowMajor;
  else if (kernel_name == "Strided")
    kernel = MatMulKernelStrided;
  else {
    printf("Invalid Kernel. Possible Kernel:\n"
           "\t 0. CuBLAS\n"
           "\t 1. Naive\n"
           "\t 2. RowMajor\n"
           "\t 3. FMA\n"
           "\t 4. Strided\n"
           );
    exit(-1);
  }

  return kernel;

}

void getDeviceInfo() {
  int device;
  cudaGetDevice(&device);
  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, device);

  std::cout << "Device Name: " << props.name << std::endl;
  std::cout << "Compute Capability: " << props.major << props.minor << std::endl;
  std::cout << "Number of SMs: " << props.multiProcessorCount << std::endl;
  std::cout << "Max Threads per SM: " << props.maxThreadsPerMultiProcessor
            << std::endl;
  std::cout << "Max Threads per Block: " << props.maxThreadsPerBlock
            << std::endl;
  std::cout << "Shared Memory per Block: " << props.sharedMemPerBlock
            << std::endl;
  std::cout << "Wraps per Block: " << ceil((THREADS * THREADS) / 32)
            << std::endl;
  return;
};

