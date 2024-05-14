#include "functional"
#include "matmul.h"
#include "matrix.h"
#include "runner.h"

using namespace std;


#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

void cudaCheck1(cudaError_t error, const char *file, int line) {
  if (error != cudaSuccess) {
    printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
           cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
};

#define cudaCheck(err) (cudaCheck1(err, __FILE__, __LINE__))

function<void()> runner(cublasHandle_t handle, MatMulKernel kernel,
                             HostMatrices host, DeviceMatrices device,
                             size_t N) {

  if (kernel == MatMulKernelCuBLAS) {
    return ([handle, device, N]() {
      cuBlas_MatMul(handle, device.dA, device.dB, device.dC, N);
      cudaCheck(cudaDeviceSynchronize());
    });
  }

  dim3 blockDim(THREADS, THREADS);
  dim3 gridDim(CEIL_DIV(N, THREADS), CEIL_DIV(N, THREADS));

  if (kernel == MatMulKernelNaive) {
    return ([handle, device, N, blockDim, gridDim]() {
      MatMulKernel_Naive<<<gridDim, blockDim>>>(device.dA, device.dB, device.dC, N);
      cudaCheck(cudaDeviceSynchronize());
    });
  }

  cudaMemcpy(device.dB, host.hB, matrix_size(host.hB, N),
             cudaMemcpyHostToDevice);

  std::function<void()> func;

  if (kernel == MatMulKernelStrided)
    func = ([handle, device, N, blockDim, gridDim]() {
      MatMulKernel_Strided<<<gridDim, blockDim>>>(device.dA, device.dB,
                                                        device.dC, N);
      cudaCheck(cudaDeviceSynchronize());
    });

  return func;
}

std::string matmulKernelToString(MatMulKernel kernel) {

  switch (kernel) {
  case MatMulKernel::MatMulKernelCuBLAS:
    return "MatMulKernelCuBLAS";
  case MatMulKernel::MatMulKernelNaive:
    return "MatMulKernelNaive";
  case MatMulKernel::MatMulKernelStrided:
    return "MatMulKernelStrided";
  default:
    return "UNKNOWN";
  }
}
