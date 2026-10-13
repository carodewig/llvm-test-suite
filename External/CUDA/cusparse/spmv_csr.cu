//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// cuSPARSE's generic sparse matrix-vector product,
// y = alpha * op(A) * x + beta * y, for a matrix in CSR format, with and
// without transposing it, on small integers whose results are exact. The host
// computes what to expect from the same matrix written out densely.

#include <cusparse.h>

#include "cuda_library_test.h"

#define CHECK_CUSPARSE(call) CHECK_STATUS(call, CUSPARSE_STATUS_SUCCESS)

// A is rows x cols.
const int rows = 4;
const int cols = 5;
const double dense[rows][cols] = {
    {1, 0, 0, 2, 0},
    {0, 0, 3, 0, 0},
    {4, 5, 0, 0, -6},
    {0, 0, 0, 0, 7},
};
const int nnz = 7;

static void test_spmv(cusparseHandle_t handle, cusparseSpMatDescr_t a,
                      const char *name, bool transpose) {
  int in = transpose ? rows : cols;
  int out = transpose ? cols : rows;
  float x[cols], y[cols], expected[cols];
  for (int i = 0; i < in; ++i)
    x[i] = i - 2;
  for (int i = 0; i < out; ++i)
    y[i] = i + 1;
  const float alpha = 2, beta = -1;
  for (int i = 0; i < out; ++i) {
    float sum = 0;
    for (int j = 0; j < in; ++j)
      sum += (transpose ? dense[j][i] : dense[i][j]) * x[j];
    expected[i] = alpha * sum + beta * y[i];
  }

  float *dx, *dy;
  CHECK_CUDA(cudaMalloc(&dx, in * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dy, out * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(dx, x, in * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dy, y, out * sizeof(float), cudaMemcpyHostToDevice));
  cusparseDnVecDescr_t vx, vy;
  CHECK_CUSPARSE(cusparseCreateDnVec(&vx, in, dx, CUDA_R_32F));
  CHECK_CUSPARSE(cusparseCreateDnVec(&vy, out, dy, CUDA_R_32F));

  cusparseOperation_t op = transpose ? CUSPARSE_OPERATION_TRANSPOSE
                                     : CUSPARSE_OPERATION_NON_TRANSPOSE;
  size_t buffer_size;
  CHECK_CUSPARSE(cusparseSpMV_bufferSize(handle, op, &alpha, a, vx, &beta, vy,
                                         CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT,
                                         &buffer_size));
  void *dbuffer;
  CHECK_CUDA(cudaMalloc(&dbuffer, buffer_size));
  CHECK_CUSPARSE(cusparseSpMV(handle, op, &alpha, a, vx, &beta, vy, CUDA_R_32F,
                              CUSPARSE_SPMV_ALG_DEFAULT, dbuffer));
  CHECK_CUDA(cudaMemcpy(y, dy, out * sizeof(float), cudaMemcpyDeviceToHost));
  check_all_close(name, y, expected, out, 0);

  CHECK_CUSPARSE(cusparseDestroyDnVec(vx));
  CHECK_CUSPARSE(cusparseDestroyDnVec(vy));
  CHECK_CUDA(cudaFree(dbuffer));
  CHECK_CUDA(cudaFree(dx));
  CHECK_CUDA(cudaFree(dy));
}

int main() {
  // The CSR form of the dense matrix.
  int offsets[rows + 1], columns[nnz];
  float values[nnz];
  int count = 0;
  for (int i = 0; i < rows; ++i) {
    offsets[i] = count;
    for (int j = 0; j < cols; ++j) {
      if (dense[i][j] != 0) {
        columns[count] = j;
        values[count] = dense[i][j];
        ++count;
      }
    }
  }
  offsets[rows] = count;
  check("CSR conversion", count == nnz);

  int *doffsets, *dcolumns;
  float *dvalues;
  CHECK_CUDA(cudaMalloc(&doffsets, sizeof(offsets)));
  CHECK_CUDA(cudaMalloc(&dcolumns, sizeof(columns)));
  CHECK_CUDA(cudaMalloc(&dvalues, sizeof(values)));
  CHECK_CUDA(cudaMemcpy(doffsets, offsets, sizeof(offsets),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dcolumns, columns, sizeof(columns),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dvalues, values, sizeof(values),
                        cudaMemcpyHostToDevice));

  cusparseHandle_t handle;
  CHECK_CUSPARSE(cusparseCreate(&handle));
  cusparseSpMatDescr_t a;
  CHECK_CUSPARSE(cusparseCreateCsr(&a, rows, cols, nnz, doffsets, dcolumns,
                                   dvalues, CUSPARSE_INDEX_32I,
                                   CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                                   CUDA_R_32F));
  test_spmv(handle, a, "cusparseSpMV", false);
  test_spmv(handle, a, "cusparseSpMV transposed", true);

  CHECK_CUSPARSE(cusparseDestroySpMat(a));
  CHECK_CUSPARSE(cusparseDestroy(handle));
  CHECK_CUDA(cudaFree(doffsets));
  CHECK_CUDA(cudaFree(dcolumns));
  CHECK_CUDA(cudaFree(dvalues));
  return test_result();
}
