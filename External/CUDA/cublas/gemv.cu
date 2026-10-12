//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// cuBLAS matrix-vector products, y = alpha * op(A) * x + beta * y, with and
// without transposing A, on small integers whose results are exact.

#include <cublas_v2.h>

#include "cuda_library_test.h"

#define CHECK_CUBLAS(call) CHECK_STATUS(call, CUBLAS_STATUS_SUCCESS)

// A is m x n, column-major, as cuBLAS expects.
const int m = 3;
const int n = 4;

// y = alpha * op(A) * x + beta * y on the host, where op(A) is A or its
// transpose.
static void host_gemv(bool transpose, float alpha, const float *a,
                      const float *x, float beta, float *y) {
  int rows = transpose ? n : m;
  int cols = transpose ? m : n;
  for (int i = 0; i < rows; ++i) {
    float sum = 0;
    for (int j = 0; j < cols; ++j)
      sum += (transpose ? a[i * m + j] : a[j * m + i]) * x[j];
    y[i] = alpha * sum + beta * y[i];
  }
}

static void test_gemv(cublasHandle_t handle, const char *name, bool transpose,
                      const float *da, const float *a) {
  // op(A) is rows x cols.
  int rows = transpose ? n : m;
  int cols = transpose ? m : n;
  float x[n], y[n], expected[n];
  for (int i = 0; i < cols; ++i)
    x[i] = i - 1;
  for (int i = 0; i < rows; ++i)
    y[i] = expected[i] = 2 * i + 1;
  const float alpha = 2, beta = -3;
  host_gemv(transpose, alpha, a, x, beta, expected);

  float *dx, *dy;
  CHECK_CUDA(cudaMalloc(&dx, cols * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dy, rows * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(dx, x, cols * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dy, y, rows * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUBLAS(cublasSgemv(handle, transpose ? CUBLAS_OP_T : CUBLAS_OP_N, m,
                           n, &alpha, da, m, dx, 1, &beta, dy, 1));
  CHECK_CUDA(cudaMemcpy(y, dy, rows * sizeof(float), cudaMemcpyDeviceToHost));
  check_all_close(name, y, expected, rows, 0);
  CHECK_CUDA(cudaFree(dx));
  CHECK_CUDA(cudaFree(dy));
}

int main() {
  float a[m * n];
  for (int i = 0; i < m * n; ++i)
    a[i] = (i * 7) % 5 - 2;

  float *da;
  CHECK_CUDA(cudaMalloc(&da, m * n * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(da, a, m * n * sizeof(float), cudaMemcpyHostToDevice));

  cublasHandle_t handle;
  CHECK_CUBLAS(cublasCreate(&handle));
  test_gemv(handle, "cublasSgemv", false, da, a);
  test_gemv(handle, "cublasSgemv transposed", true, da, a);
  CHECK_CUBLAS(cublasDestroy(handle));

  CHECK_CUDA(cudaFree(da));
  return test_result();
}
