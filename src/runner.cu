#include "functional"
#include "matmul.h"
#include "matrix.h"
#include "runner.h"

using namespace std;


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

  dim3 threadsPerBlock(THREADS, THREADS);
  dim3 blocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
              (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

  if (kernel == MatMulKernelNaive) {
    return ([handle, device, N, threadsPerBlock, blocks]() {
      MatMulKernel_Naive<<<blocks, threadsPerBlock>>>(device.dA, device.dB, device.dC, N);
      cudaCheck(cudaDeviceSynchronize());
    });
  }

  transpose(host.hB, N);
  cudaMemcpy(device.dB, host.hB, matrix_size(host.hB, N),
             cudaMemcpyHostToDevice);

  std::function<void()> func;

  if (kernel == MatMulKernelRowMajor)
    func = ([handle, device, N, threadsPerBlock, blocks]() {
      MatMulKernel_RowMajor<<<blocks, threadsPerBlock>>>(device.dA, device.dB,
                                                         device.dC, N);
      cudaCheck(cudaDeviceSynchronize());
    });

  if (kernel == MatMulKernelStrided)
    func = ([handle, device, N, threadsPerBlock, blocks]() {
      MatMulKernel_Strided<<<blocks, threadsPerBlock>>>(device.dA, device.dB,
                                                        device.dC, N);
      cudaCheck(cudaDeviceSynchronize());
    });

  transpose(host.hB, N);
  return func;
}

std::string matmulKernelToString(MatMulKernel kernel) {

  switch (kernel) {
  case MatMulKernel::MatMulKernelCuBLAS:
    return "MatMulKernelCuBLAS";
  case MatMulKernel::MatMulKernelNaive:
    return "MatMulKernelNaive";
  case MatMulKernel::MatMulKernelRowMajor:
    return "MatMulKernelRowMajor";
  case MatMulKernel::MatMulKernelStrided:
    return "MatMulKernelStrided";
  default:
    return "UNKNOWN";
  }
}
