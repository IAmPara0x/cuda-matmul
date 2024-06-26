#include <string>
#ifndef RUNNER_H

#define RUNNER_H 1

#include <cstddef>
#include <cublas_v2.h>
#include <functional>

typedef enum MatMulKernel {
  MatMulKernelCuBLAS,
  MatMulKernelNaive,
  MatMulKernelGlobalCoalesce,
  MatMulKernelStrided,
  MatMulKernel1DBlockTiling,
  MatMulKernel2DBlockTiling,
  MatMulKernelFinal,
} MatMulKernel;


typedef struct HostMatrices {
  float *hA;
  float *hB;
  float *hC;
} HostMatrices;

typedef struct DeviceMatrices {
  float *dA;
  float *dB;
  float *dC;
} DeviceMatrices;

std::string matmulKernelToString(MatMulKernel kernel);
std::function<void()> runner(cublasHandle_t handle, MatMulKernel kernel,
                             HostMatrices host, DeviceMatrices device,
                             size_t N);

#endif

