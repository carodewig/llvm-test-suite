//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// cuBLAS level-1 routines on vectors whose results are known exactly.

#include <cublas_v2.h>

#include "cuda_library_test.h"

#define CHECK_CUBLAS(call) CHECK_STATUS(call, CUBLAS_STATUS_SUCCESS)

int main() {
  const int n = 8;
  float x[n], y[n];
  for (int i = 0; i < n; ++i) {
    x[i] = i + 1;
    y[i] = n - i;
  }
  // The element of x with the largest magnitude, which isamax finds.
  x[5] = -10;

  float *dx, *dy;
  CHECK_CUDA(cudaMalloc(&dx, n * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dy, n * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(dx, x, n * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dy, y, n * sizeof(float), cudaMemcpyHostToDevice));

  cublasHandle_t handle;
  CHECK_CUBLAS(cublasCreate(&handle));

  double expected_dot = 0, expected_sum_squares = 0;
  for (int i = 0; i < n; ++i) {
    expected_dot += x[i] * y[i];
    expected_sum_squares += x[i] * x[i];
  }

  float dot;
  CHECK_CUBLAS(cublasSdot(handle, n, dx, 1, dy, 1, &dot));
  check_close("cublasSdot", dot, expected_dot, 0);

  float norm;
  CHECK_CUBLAS(cublasSnrm2(handle, n, dx, 1, &norm));
  check_close("cublasSnrm2", norm, sqrt(expected_sum_squares), 1e-6);

  // A 1-based index.
  int max_index;
  CHECK_CUBLAS(cublasIsamax(handle, n, dx, 1, &max_index));
  check("cublasIsamax", max_index == 6);

  const float alpha = 2;
  float axpy[n], expected_axpy[n];
  for (int i = 0; i < n; ++i)
    expected_axpy[i] = alpha * x[i] + y[i];
  CHECK_CUBLAS(cublasSaxpy(handle, n, &alpha, dx, 1, dy, 1));
  CHECK_CUDA(cudaMemcpy(axpy, dy, n * sizeof(float), cudaMemcpyDeviceToHost));
  check_all_close("cublasSaxpy", axpy, expected_axpy, n, 0);

  CHECK_CUBLAS(cublasDestroy(handle));
  CHECK_CUDA(cudaFree(dx));
  CHECK_CUDA(cudaFree(dy));
  return test_result();
}
