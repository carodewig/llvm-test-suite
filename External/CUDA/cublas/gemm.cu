//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// cuBLAS matrix products, C = alpha * op(A) * op(B) + beta * C, in single and
// double precision and with each operand transposed or not, on small integers
// whose results are exact.

#include <cublas_v2.h>

#include "cuda_library_test.h"

#define CHECK_CUBLAS(call) CHECK_STATUS(call, CUBLAS_STATUS_SUCCESS)

// op(A) is m x k, op(B) is k x n and C is m x n, all column-major.
const int m = 3;
const int n = 2;
const int k = 4;

static cublasStatus_t gemm(cublasHandle_t handle, cublasOperation_t op_a,
                           cublasOperation_t op_b, const float *alpha,
                           const float *a, int lda, const float *b, int ldb,
                           const float *beta, float *c) {
  return cublasSgemm(handle, op_a, op_b, m, n, k, alpha, a, lda, b, ldb, beta,
                     c, m);
}

static cublasStatus_t gemm(cublasHandle_t handle, cublasOperation_t op_a,
                           cublasOperation_t op_b, const double *alpha,
                           const double *a, int lda, const double *b, int ldb,
                           const double *beta, double *c) {
  return cublasDgemm(handle, op_a, op_b, m, n, k, alpha, a, lda, b, ldb, beta,
                     c, m);
}

template <typename T>
static void test_gemm(cublasHandle_t handle, const char *name,
                      bool transpose_a, bool transpose_b) {
  // The stored matrices: A is m x k, or k x m when transposed, and the same
  // for B.
  int lda = transpose_a ? k : m;
  int ldb = transpose_b ? n : k;
  T a[m * k], b[k * n], c[m * n], expected[m * n];
  for (int i = 0; i < m * k; ++i)
    a[i] = (i * 5) % 7 - 3;
  for (int i = 0; i < k * n; ++i)
    b[i] = (i * 3) % 4 - 1;
  for (int i = 0; i < m * n; ++i)
    c[i] = i;
  const T alpha = 2, beta = -1;

  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      T sum = 0;
      for (int l = 0; l < k; ++l) {
        T a_il = transpose_a ? a[i * lda + l] : a[l * lda + i];
        T b_lj = transpose_b ? b[l * ldb + j] : b[j * ldb + l];
        sum += a_il * b_lj;
      }
      expected[j * m + i] = alpha * sum + beta * c[j * m + i];
    }
  }

  T *da, *db, *dc;
  CHECK_CUDA(cudaMalloc(&da, sizeof(a)));
  CHECK_CUDA(cudaMalloc(&db, sizeof(b)));
  CHECK_CUDA(cudaMalloc(&dc, sizeof(c)));
  CHECK_CUDA(cudaMemcpy(da, a, sizeof(a), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(db, b, sizeof(b), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dc, c, sizeof(c), cudaMemcpyHostToDevice));
  CHECK_CUBLAS(gemm(handle, transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N,
                    transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N, &alpha, da, lda,
                    db, ldb, &beta, dc));
  CHECK_CUDA(cudaMemcpy(c, dc, sizeof(c), cudaMemcpyDeviceToHost));
  check_all_close(name, c, expected, m * n, 0);
  CHECK_CUDA(cudaFree(da));
  CHECK_CUDA(cudaFree(db));
  CHECK_CUDA(cudaFree(dc));
}

int main() {
  cublasHandle_t handle;
  CHECK_CUBLAS(cublasCreate(&handle));
  test_gemm<float>(handle, "cublasSgemm", false, false);
  test_gemm<float>(handle, "cublasSgemm A transposed", true, false);
  test_gemm<float>(handle, "cublasSgemm B transposed", false, true);
  test_gemm<double>(handle, "cublasDgemm", false, false);
  test_gemm<double>(handle, "cublasDgemm both transposed", true, true);
  CHECK_CUBLAS(cublasDestroy(handle));
  return test_result();
}
