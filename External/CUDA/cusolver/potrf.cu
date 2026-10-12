//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// cuSOLVER's Cholesky factorization of a symmetric positive definite matrix
// A = L * L^T, built from a known lower triangular L of small integers, and the
// solution of A * x = b with that factorization, for a b computed from a known
// x.

#include <cusolverDn.h>

#include "cuda_library_test.h"

#define CHECK_CUSOLVER(call) CHECK_STATUS(call, CUSOLVER_STATUS_SUCCESS)

const int n = 4;

int main() {
  // Column-major, as cuSOLVER expects: element (i, j) is at j * n + i.
  const double l_rows[n][n] = {
      {2, 0, 0, 0},
      {1, 3, 0, 0},
      {-1, 2, 1, 0},
      {0, 1, -2, 4},
  };
  const double x[n] = {1, -2, 3, 0.5};
  double l[n * n], a[n * n], b[n];
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      l[j * n + i] = l_rows[i][j];
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      double sum = 0;
      for (int k = 0; k < n; ++k)
        sum += l_rows[i][k] * l_rows[j][k];
      a[j * n + i] = sum;
    }
  }
  for (int i = 0; i < n; ++i) {
    b[i] = 0;
    for (int j = 0; j < n; ++j)
      b[i] += a[j * n + i] * x[j];
  }

  cusolverDnHandle_t handle;
  CHECK_CUSOLVER(cusolverDnCreate(&handle));
  double *da, *db, *dwork;
  int *dinfo;
  CHECK_CUDA(cudaMalloc(&da, sizeof(a)));
  CHECK_CUDA(cudaMalloc(&db, sizeof(b)));
  CHECK_CUDA(cudaMalloc(&dinfo, sizeof(int)));
  CHECK_CUDA(cudaMemcpy(da, a, sizeof(a), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(db, b, sizeof(b), cudaMemcpyHostToDevice));

  int lwork;
  CHECK_CUSOLVER(cusolverDnDpotrf_bufferSize(handle, CUBLAS_FILL_MODE_LOWER, n,
                                             da, n, &lwork));
  CHECK_CUDA(cudaMalloc(&dwork, lwork * sizeof(double)));
  CHECK_CUSOLVER(cusolverDnDpotrf(handle, CUBLAS_FILL_MODE_LOWER, n, da, n,
                                  dwork, lwork, dinfo));
  int info;
  CHECK_CUDA(cudaMemcpy(&info, dinfo, sizeof(int), cudaMemcpyDeviceToHost));
  check("cusolverDnDpotrf info", info == 0);

  // potrf leaves the upper triangle as it was, so only the lower one is L.
  double factor[n * n];
  CHECK_CUDA(cudaMemcpy(factor, da, sizeof(factor), cudaMemcpyDeviceToHost));
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < j; ++i)
      factor[j * n + i] = 0;
  check_all_close("cusolverDnDpotrf factor", factor, l, n * n, 1e-12);

  CHECK_CUSOLVER(cusolverDnDpotrs(handle, CUBLAS_FILL_MODE_LOWER, n, 1, da, n,
                                  db, n, dinfo));
  CHECK_CUDA(cudaMemcpy(&info, dinfo, sizeof(int), cudaMemcpyDeviceToHost));
  check("cusolverDnDpotrs info", info == 0);
  double solution[n];
  CHECK_CUDA(cudaMemcpy(solution, db, sizeof(solution),
                        cudaMemcpyDeviceToHost));
  check_all_close("cusolverDnDpotrs solution", solution, x, n, 1e-12);

  CHECK_CUSOLVER(cusolverDnDestroy(handle));
  CHECK_CUDA(cudaFree(da));
  CHECK_CUDA(cudaFree(db));
  CHECK_CUDA(cudaFree(dwork));
  CHECK_CUDA(cudaFree(dinfo));
  return test_result();
}
