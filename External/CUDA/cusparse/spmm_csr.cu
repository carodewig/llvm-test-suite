//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// cuSPARSE's generic product of a sparse matrix in CSR format and a dense
// matrix, C = alpha * A * B + beta * C, with the dense matrices stored
// column-major and row-major, on small integers whose results are exact. The
// host computes what to expect from the sparse matrix written out densely.

#include <cusparse.h>

#include "cuda_library_test.h"

#define CHECK_CUSPARSE(call) CHECK_STATUS(call, CUSPARSE_STATUS_SUCCESS)

// A is rows x inner, B is inner x cols and C is rows x cols.
const int rows = 4;
const int inner = 5;
const int cols = 3;
const double dense[rows][inner] = {
    {1, 0, 0, 2, 0},
    {0, 0, 3, 0, 0},
    {4, 5, 0, 0, -6},
    {0, 0, 0, 0, 7},
};
const int nnz = 7;

// Element (i, j) of a matrix with 'n_rows' rows and 'n_cols' columns.
static int at(bool row_major, int i, int j, int n_rows, int n_cols) {
  return row_major ? i * n_cols + j : j * n_rows + i;
}

static void test_spmm(cusparseHandle_t handle, cusparseSpMatDescr_t a,
                      const char *name, bool row_major) {
  float b[inner * cols], c[rows * cols], expected[rows * cols];
  for (int i = 0; i < inner; ++i)
    for (int j = 0; j < cols; ++j)
      b[at(row_major, i, j, inner, cols)] = (i * 3 + j * 5) % 7 - 3;
  for (int i = 0; i < rows; ++i)
    for (int j = 0; j < cols; ++j)
      c[at(row_major, i, j, rows, cols)] = i - j;
  const float alpha = 2, beta = -1;
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      float sum = 0;
      for (int k = 0; k < inner; ++k)
        sum += dense[i][k] * b[at(row_major, k, j, inner, cols)];
      int ij = at(row_major, i, j, rows, cols);
      expected[ij] = alpha * sum + beta * c[ij];
    }
  }

  float *db, *dc;
  CHECK_CUDA(cudaMalloc(&db, sizeof(b)));
  CHECK_CUDA(cudaMalloc(&dc, sizeof(c)));
  CHECK_CUDA(cudaMemcpy(db, b, sizeof(b), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dc, c, sizeof(c), cudaMemcpyHostToDevice));
  cusparseOrder_t order = row_major ? CUSPARSE_ORDER_ROW : CUSPARSE_ORDER_COL;
  cusparseDnMatDescr_t mb, mc;
  CHECK_CUSPARSE(cusparseCreateDnMat(&mb, inner, cols,
                                     row_major ? cols : inner, db, CUDA_R_32F,
                                     order));
  CHECK_CUSPARSE(cusparseCreateDnMat(&mc, rows, cols, row_major ? cols : rows,
                                     dc, CUDA_R_32F, order));

  cusparseOperation_t op = CUSPARSE_OPERATION_NON_TRANSPOSE;
  size_t buffer_size;
  CHECK_CUSPARSE(cusparseSpMM_bufferSize(handle, op, op, &alpha, a, mb, &beta,
                                         mc, CUDA_R_32F,
                                         CUSPARSE_SPMM_ALG_DEFAULT,
                                         &buffer_size));
  void *dbuffer;
  CHECK_CUDA(cudaMalloc(&dbuffer, buffer_size));
  CHECK_CUSPARSE(cusparseSpMM(handle, op, op, &alpha, a, mb, &beta, mc,
                              CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, dbuffer));
  CHECK_CUDA(cudaMemcpy(c, dc, sizeof(c), cudaMemcpyDeviceToHost));
  check_all_close(name, c, expected, rows * cols, 0);

  CHECK_CUSPARSE(cusparseDestroyDnMat(mb));
  CHECK_CUSPARSE(cusparseDestroyDnMat(mc));
  CHECK_CUDA(cudaFree(dbuffer));
  CHECK_CUDA(cudaFree(db));
  CHECK_CUDA(cudaFree(dc));
}

int main() {
  // The CSR form of the dense matrix.
  int offsets[rows + 1], columns[nnz];
  float values[nnz];
  int count = 0;
  for (int i = 0; i < rows; ++i) {
    offsets[i] = count;
    for (int j = 0; j < inner; ++j) {
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
  CHECK_CUSPARSE(cusparseCreateCsr(&a, rows, inner, nnz, doffsets, dcolumns,
                                   dvalues, CUSPARSE_INDEX_32I,
                                   CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                                   CUDA_R_32F));
  test_spmm(handle, a, "cusparseSpMM column-major", false);
  test_spmm(handle, a, "cusparseSpMM row-major", true);

  CHECK_CUSPARSE(cusparseDestroySpMat(a));
  CHECK_CUSPARSE(cusparseDestroy(handle));
  CHECK_CUDA(cudaFree(doffsets));
  CHECK_CUDA(cudaFree(dcolumns));
  CHECK_CUDA(cudaFree(dvalues));
  return test_result();
}
