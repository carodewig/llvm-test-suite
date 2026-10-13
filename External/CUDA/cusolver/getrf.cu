//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// cuSOLVER's LU factorization with partial pivoting, P * A = L * U, of a matrix
// whose first pivot is 0, so that it has to swap rows. The factors and the row
// swaps it returns must multiply back to A, and solving A * x = b with them
// must give the known x that b was computed from.

#include <cusolverDn.h>

#include "cuda_library_test.h"

#define CHECK_CUSOLVER(call) CHECK_STATUS(call, CUSOLVER_STATUS_SUCCESS)

const int n = 4;

int main() {
  const double a_rows[n][n] = {
      {0, 2, -1, 3},
      {4, 1, 0, -2},
      {-2, 3, 5, 1},
      {1, 0, 2, 6},
  };
  const double x[n] = {2, -1, 0.5, 3};
  // Column-major, as cuSOLVER expects: element (i, j) is at j * n + i.
  double a[n * n], b[n];
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      a[j * n + i] = a_rows[i][j];
  for (int i = 0; i < n; ++i) {
    b[i] = 0;
    for (int j = 0; j < n; ++j)
      b[i] += a_rows[i][j] * x[j];
  }

  cusolverDnHandle_t handle;
  CHECK_CUSOLVER(cusolverDnCreate(&handle));
  double *da, *db, *dwork;
  int *dpivots, *dinfo;
  CHECK_CUDA(cudaMalloc(&da, sizeof(a)));
  CHECK_CUDA(cudaMalloc(&db, sizeof(b)));
  CHECK_CUDA(cudaMalloc(&dpivots, n * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&dinfo, sizeof(int)));
  CHECK_CUDA(cudaMemcpy(da, a, sizeof(a), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(db, b, sizeof(b), cudaMemcpyHostToDevice));

  int lwork;
  CHECK_CUSOLVER(cusolverDnDgetrf_bufferSize(handle, n, n, da, n, &lwork));
  CHECK_CUDA(cudaMalloc(&dwork, lwork * sizeof(double)));
  CHECK_CUSOLVER(cusolverDnDgetrf(handle, n, n, da, n, dwork, dpivots, dinfo));
  int info;
  CHECK_CUDA(cudaMemcpy(&info, dinfo, sizeof(int), cudaMemcpyDeviceToHost));
  check("cusolverDnDgetrf info", info == 0);

  // L has a unit diagonal and is stored below it, U on and above it. Row i was
  // swapped with row pivots[i] - 1, in order.
  double lu[n * n];
  int pivots[n];
  CHECK_CUDA(cudaMemcpy(lu, da, sizeof(lu), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(pivots, dpivots, sizeof(pivots),
                        cudaMemcpyDeviceToHost));
  check("cusolverDnDgetrf swaps the first row", pivots[0] != 1);
  double product[n * n];
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      double sum = 0;
      for (int k = 0; k <= i && k <= j; ++k)
        sum += (k == i ? 1 : lu[k * n + i]) * lu[j * n + k];
      product[j * n + i] = sum;
    }
  }
  // Undo the row swaps, last first, to get back to A.
  for (int i = n - 1; i >= 0; --i) {
    int other = pivots[i] - 1;
    for (int j = 0; j < n; ++j) {
      double t = product[j * n + i];
      product[j * n + i] = product[j * n + other];
      product[j * n + other] = t;
    }
  }
  check_all_close("cusolverDnDgetrf factors", product, a, n * n, 1e-12);

  CHECK_CUSOLVER(cusolverDnDgetrs(handle, CUBLAS_OP_N, n, 1, da, n, dpivots,
                                  db, n, dinfo));
  CHECK_CUDA(cudaMemcpy(&info, dinfo, sizeof(int), cudaMemcpyDeviceToHost));
  check("cusolverDnDgetrs info", info == 0);
  double solution[n];
  CHECK_CUDA(cudaMemcpy(solution, db, sizeof(solution),
                        cudaMemcpyDeviceToHost));
  check_all_close("cusolverDnDgetrs solution", solution, x, n, 1e-12);

  CHECK_CUSOLVER(cusolverDnDestroy(handle));
  CHECK_CUDA(cudaFree(da));
  CHECK_CUDA(cudaFree(db));
  CHECK_CUDA(cudaFree(dwork));
  CHECK_CUDA(cudaFree(dpivots));
  CHECK_CUDA(cudaFree(dinfo));
  return test_result();
}
