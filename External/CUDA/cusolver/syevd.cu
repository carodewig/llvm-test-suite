//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// cuSOLVER's eigendecomposition of a symmetric matrix with a known spectrum,
// A = H * D * H, where D is diagonal and H a Householder reflection, which is
// symmetric and orthogonal. The eigenvalues must be those of D, in ascending
// order, and each eigenvector v it returns must satisfy A * v = lambda * v,
// whichever sign it has.

#include <cusolverDn.h>

#include "cuda_library_test.h"

#define CHECK_CUSOLVER(call) CHECK_STATUS(call, CUSOLVER_STATUS_SUCCESS)

const int n = 4;

int main() {
  // In ascending order, as syevd returns them.
  const double eigenvalues[n] = {-3, 1, 2, 5};
  // H = I - 2 * v * v^T / (v^T * v).
  const double v[n] = {1, 2, -1, 3};
  double vv = 0;
  for (int i = 0; i < n; ++i)
    vv += v[i] * v[i];
  double h[n][n];
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      h[i][j] = (i == j) - 2 * v[i] * v[j] / vv;
  // Column-major, as cuSOLVER expects: element (i, j) is at j * n + i. The
  // eigenvalues are put on the diagonal out of order, so that sorting them is
  // part of the test.
  const int order[n] = {2, 0, 3, 1};
  double a[n * n];
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      double sum = 0;
      for (int k = 0; k < n; ++k)
        sum += h[i][k] * eigenvalues[order[k]] * h[k][j];
      a[j * n + i] = sum;
    }
  }

  cusolverDnHandle_t handle;
  CHECK_CUSOLVER(cusolverDnCreate(&handle));
  double *da, *dw, *dwork;
  int *dinfo;
  CHECK_CUDA(cudaMalloc(&da, sizeof(a)));
  CHECK_CUDA(cudaMalloc(&dw, n * sizeof(double)));
  CHECK_CUDA(cudaMalloc(&dinfo, sizeof(int)));
  CHECK_CUDA(cudaMemcpy(da, a, sizeof(a), cudaMemcpyHostToDevice));

  int lwork;
  CHECK_CUSOLVER(cusolverDnDsyevd_bufferSize(handle, CUSOLVER_EIG_MODE_VECTOR,
                                             CUBLAS_FILL_MODE_LOWER, n, da, n,
                                             dw, &lwork));
  CHECK_CUDA(cudaMalloc(&dwork, lwork * sizeof(double)));
  CHECK_CUSOLVER(cusolverDnDsyevd(handle, CUSOLVER_EIG_MODE_VECTOR,
                                  CUBLAS_FILL_MODE_LOWER, n, da, n, dw, dwork,
                                  lwork, dinfo));
  int info;
  CHECK_CUDA(cudaMemcpy(&info, dinfo, sizeof(int), cudaMemcpyDeviceToHost));
  check("cusolverDnDsyevd info", info == 0);

  double w[n], vectors[n * n];
  CHECK_CUDA(cudaMemcpy(w, dw, sizeof(w), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(vectors, da, sizeof(vectors), cudaMemcpyDeviceToHost));
  check_all_close("cusolverDnDsyevd eigenvalues", w, eigenvalues, n, 1e-12);

  // A * v - lambda * v for each eigenvector, column j of the result, and the
  // length of each, which must be 1.
  double residuals[n * n], zeros[n * n], lengths[n], ones[n];
  for (int j = 0; j < n; ++j) {
    lengths[j] = 0;
    ones[j] = 1;
    for (int i = 0; i < n; ++i) {
      double sum = 0;
      for (int k = 0; k < n; ++k)
        sum += a[k * n + i] * vectors[j * n + k];
      residuals[j * n + i] = sum - w[j] * vectors[j * n + i];
      zeros[j * n + i] = 0;
      lengths[j] += vectors[j * n + i] * vectors[j * n + i];
    }
  }
  check_all_close("cusolverDnDsyevd eigenvectors", residuals, zeros, n * n,
                  1e-12);
  check_all_close("cusolverDnDsyevd eigenvector lengths", lengths, ones, n,
                  1e-12);

  CHECK_CUSOLVER(cusolverDnDestroy(handle));
  CHECK_CUDA(cudaFree(da));
  CHECK_CUDA(cudaFree(dw));
  CHECK_CUDA(cudaFree(dwork));
  CHECK_CUDA(cudaFree(dinfo));
  return test_result();
}
