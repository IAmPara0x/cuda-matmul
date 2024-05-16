#include "functional"
#include "matmul.cuh"
#include "runner.h"
#include <stdexcept>

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


  if (kernel == MatMulKernelNaive) {

    dim3 blockDim(16, 16);
    dim3 gridDim(CEIL_DIV(N, 16), CEIL_DIV(N, 16));

    return ([handle, device, N, blockDim, gridDim]() {
      MatMulKernel_Naive<<<gridDim, blockDim>>>(device.dA, device.dB, device.dC, N);
      cudaCheck(cudaDeviceSynchronize());
    });

  }

  if (kernel == MatMulKernelGlobalCoalesce) {

    dim3 blockDim(16 * 16);
    dim3 gridDim(CEIL_DIV(N, 16), CEIL_DIV(N, 16));

    return ([handle, device, N, blockDim, gridDim]() {
      MatMulKernel_GlobalCoalesce<16><<<gridDim, blockDim>>>(device.dA, device.dB, device.dC, N);
      cudaCheck(cudaDeviceSynchronize());
    });

  }

  if (kernel == MatMulKernelStrided) {


      constexpr uint T = 16;
      dim3 blockDim(T * T);
      dim3 gridDim(CEIL_DIV(N, T), CEIL_DIV(N, T));

      return ([handle, device, N, blockDim, gridDim]() { 
          MatMulKernel_Strided<T>
            <<<gridDim, blockDim>>>(device.dA, device.dB, device.dC, N);
          cudaCheck(cudaDeviceSynchronize());
        });
  }

  if (kernel == MatMulKernel1DBlockTiling)
  {

    constexpr uint DM = 32;
    constexpr uint DK = 8;
    constexpr uint T  = DM / DK;
    dim3 blockDim(DM, DK);
    dim3 gridDim(CEIL_DIV(N, DM), CEIL_DIV(N, DM));

    return ([handle, device, N, blockDim, gridDim]() {
      MatMulKernel_1DBlockTiling<DM,DK,T><<<gridDim, blockDim>>>(device.dA, device.dB,
                                                        device.dC, N);
      cudaCheck(cudaDeviceSynchronize());
    });
  }

  throw runtime_error("Unreachable!");
}

std::string matmulKernelToString(MatMulKernel kernel) {

  switch (kernel) {
  case MatMulKernel::MatMulKernelCuBLAS:
    return "MatMulKernelCuBLAS";
  case MatMulKernel::MatMulKernelNaive:
    return "MatMulKernelNaive";
  case MatMulKernel::MatMulKernelGlobalCoalesce:
    return "MatMulKernelGlobalCoalesce";
  case MatMulKernel::MatMulKernelStrided:
    return "MatMulKernelStrided";
  case MatMulKernel::MatMulKernel1DBlockTiling:
    return "MatMulKernel1DBlockTiling";
  default:
    return "UNKNOWN";
  }
}
