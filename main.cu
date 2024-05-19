#include "./src/matrix.h"
#include "./src/runner.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <string>

using namespace std;

constexpr int N = 1 << 12;

#define USAGE "USAGE: main <KERNEL_NAME> -iter=<NUM> -verify \n"

void getDeviceInfo();
MatMulKernel getKernelName(int argc, char **argv);

int main(int argc, char **argv) {

  MatMulKernel kernel = getKernelName(argc, argv);

  getDeviceInfo();

  cublasHandle_t handle;
  cublasCreate(&handle);

  float *hA = random_mat(N, 5), *hB = random_mat(N, 5), *hC = init_mat(N),
        *hCRef = init_mat(N);
  size_t sizeA = matrix_size(hA, N), sizeB = matrix_size(hB, N),
         sizeC = matrix_size(hC, N);
  float *dA = nullptr, *dB = nullptr, *dC = nullptr, *dCRef = nullptr;
  float gflops;

  size_t iterations = 1;

  cudaMalloc((void **)&dA, sizeA);
  cudaMalloc((void **)&dB, sizeB);
  cudaMalloc((void **)&dC, sizeC);
  cudaMalloc((void **)&dCRef, sizeC);

  cudaMemcpy(dA, hA, sizeA, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, hB, sizeB, cudaMemcpyHostToDevice);

  auto func = runner(handle, kernel, {hA, hB, hC}, {dA, dB, dC}, N);


  // HACK: Currently we only expect max 2 argument.
  if (argc == 3 && strcmp(argv[2], "-verify") == 0) {

    gflops = measure_gflops(func, N, iterations);
    printf("%s performance: (%f) GFLOPS. size: %d\n",
           matmulKernelToString(kernel).c_str(), gflops, N);
    cudaMemcpy(hC, dC, sizeC, cudaMemcpyDeviceToHost);

    runner(handle, MatMulKernelCuBLAS, {hA, hB, hCRef}, {dA, dB, dCRef}, N)();
    cudaMemcpy(hCRef, dCRef, sizeC, cudaMemcpyDeviceToHost);

    if (same_matrix(hC, hCRef, N, 1e-1))
      cout << "Verified!" << endl;
    else
      cout << "Verification Failed!" << endl;

  } else if (argc == 4 && strcmp(argv[2], "-iter") == 0) {

    // TODO: Handle error to prevent memory leak
    iterations = stoul(argv[3]);
    gflops = measure_gflops(func, N, iterations);
    printf("%s performance: (%f) GFLOPS, iterations: %zu size: %d\n",
           matmulKernelToString(kernel).c_str(), gflops, iterations, N);

  } else {
    printf("%s\n", argv[2]);
    printf(USAGE);
  }


  // cuda free, matrices:
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  cudaFree(dCRef);
  cublasDestroy(handle);

  // free matrices:
  free(hA);
  free(hB);
  free(hC);
  free(hCRef);
  cudaDeviceReset();
  return 0;
};

MatMulKernel getKernelName(int argc, char **argv) {

  if (argc == 1)
  {
    printf(USAGE);
    exit(-1);
  }

  string kernel_name = argv[1];

  MatMulKernel kernel;

  if (kernel_name == "CuBLAS")
    kernel = MatMulKernelCuBLAS;
  else if (kernel_name == "Naive")
    kernel = MatMulKernelNaive;
  else if (kernel_name == "GlobalCoalesce")
    kernel = MatMulKernelGlobalCoalesce;
  else if (kernel_name == "Strided")
    kernel = MatMulKernelStrided;
  else if (kernel_name == "1DBlockTiling")
    kernel = MatMulKernel1DBlockTiling;
  else if (kernel_name == "2DBlockTiling")
    kernel = MatMulKernel2DBlockTiling;
  else if (kernel_name == "Final")
    kernel = MatMulKernelFinal;
  else {
    printf("Invalid Kernel. Possible Kernel:\n"
           "\t 0. CuBLAS\n"
           "\t 1. Naive\n"
           "\t 2. Strided\n"
           "\t 3. 1DBlockTiling\n"
           "\t 4. 2DBlockTiling\n"
           "\t 5. Final\n"
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
  std::cout << "Max Threads per Block: " << props.maxThreadsPerBlock << std::endl;
  std::cout << "Max Threads per MultiProcessor: " << props.maxThreadsPerMultiProcessor << std::endl;
  std::cout << "Threads per Wrap: " << props.warpSize << std::endl;
  std::cout << "Max Regs per Block: " << props.regsPerBlock << std::endl;
  std::cout << "Max Regs per MultiProcessor: " << props.regsPerMultiprocessor << std::endl;
  std::cout << "Max Shared Memory per Block: " << props.sharedMemPerBlock << std::endl;
  std::cout << "Max Shared Memory per MultiProcessor: " << props.sharedMemPerMultiprocessor << std::endl;
  std::cout << "SM Count: " << props.multiProcessorCount << std::endl;
  std::cout << "Max Wrap per MultiProcessor: " << props.maxThreadsPerMultiProcessor / props.warpSize << std::endl;
  return;
};

