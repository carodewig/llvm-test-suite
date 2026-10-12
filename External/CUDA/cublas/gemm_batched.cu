//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// cuBLAS batched matrix products, C[i] = A[i] * B[i] for each matrix of a
// batch, with the batch given by a stride and by an array of pointers, on small
// integers whose results are exact.

#include <cublas_v2.h>

#include "cuda_library_test.h"

#define CHECK_CUBLAS(call) CHECK_STATUS(call, CUBLAS_STATUS_SUCCESS)

// A[i] is m x k, B[i] is k x n and C[i] is m x n, all column-major and stored
// one after the other.
const int m = 2;
const int n = 3;
const int k = 4;
const int batch = 3;
const int size_a = m * k;
const int size_b = k * n;
const int size_c = m * n;

int main() {
  float a[batch * size_a], b[batch * size_b], expected[batch * size_c];
  for (int i = 0; i < batch * size_a; ++i)
    a[i] = (i * 5) % 7 - 3;
  for (int i = 0; i < batch * size_b; ++i)
    b[i] = (i * 3) % 4 - 1;
  for (int p = 0; p < batch; ++p) {
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        float sum = 0;
        for (int l = 0; l < k; ++l)
          sum += a[p * size_a + l * m + i] * b[p * size_b + j * k + l];
        expected[p * size_c + j * m + i] = sum;
      }
    }
  }

  float *da, *db, *dc;
  CHECK_CUDA(cudaMalloc(&da, sizeof(a)));
  CHECK_CUDA(cudaMalloc(&db, sizeof(b)));
  CHECK_CUDA(cudaMalloc(&dc, sizeof(expected)));
  CHECK_CUDA(cudaMemcpy(da, a, sizeof(a), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(db, b, sizeof(b), cudaMemcpyHostToDevice));

  cublasHandle_t handle;
  CHECK_CUBLAS(cublasCreate(&handle));
  const float alpha = 1, beta = 0;
  float c[batch * size_c];

  CHECK_CUDA(cudaMemset(dc, 0, sizeof(c)));
  CHECK_CUBLAS(cublasSgemmStridedBatched(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, da, m, size_a, db, k,
      size_b, &beta, dc, m, size_c, batch));
  CHECK_CUDA(cudaMemcpy(c, dc, sizeof(c), cudaMemcpyDeviceToHost));
  check_all_close("cublasSgemmStridedBatched", c, expected, batch * size_c, 0);

  // The same batch, given as arrays of pointers to the matrices, which live on
  // the device too.
  const float *pa[batch], *pb[batch];
  float *pc[batch];
  for (int p = 0; p < batch; ++p) {
    pa[p] = da + p * size_a;
    pb[p] = db + p * size_b;
    pc[p] = dc + p * size_c;
  }
  const float **dpa, **dpb;
  float **dpc;
  CHECK_CUDA(cudaMalloc(&dpa, sizeof(pa)));
  CHECK_CUDA(cudaMalloc(&dpb, sizeof(pb)));
  CHECK_CUDA(cudaMalloc(&dpc, sizeof(pc)));
  CHECK_CUDA(cudaMemcpy(dpa, pa, sizeof(pa), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dpb, pb, sizeof(pb), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dpc, pc, sizeof(pc), cudaMemcpyHostToDevice));

  CHECK_CUDA(cudaMemset(dc, 0, sizeof(c)));
  CHECK_CUBLAS(cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                                  &alpha, dpa, m, dpb, k, &beta, dpc, m,
                                  batch));
  CHECK_CUDA(cudaMemcpy(c, dc, sizeof(c), cudaMemcpyDeviceToHost));
  check_all_close("cublasSgemmBatched", c, expected, batch * size_c, 0);

  CHECK_CUBLAS(cublasDestroy(handle));
  CHECK_CUDA(cudaFree(dpa));
  CHECK_CUDA(cudaFree(dpb));
  CHECK_CUDA(cudaFree(dpc));
  CHECK_CUDA(cudaFree(da));
  CHECK_CUDA(cudaFree(db));
  CHECK_CUDA(cudaFree(dc));
  return test_result();
}
