//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A program built by CMake with Clang as its CUDA compiler, which launches a
// kernel and calls each of the CUDA libraries once, checking what they compute.
// The CUDA runtime it runs with must be the one whose headers it was compiled
// with, which FindCUDAToolkit picked.

#include <cublas_v2.h>
#include <cufft.h>
#include <curand.h>
#include <cusolverDn.h>
#include <cusparse.h>

#include "cuda_library_test.h"

const int n = 8;

__global__ void iota(float *x) { x[threadIdx.x] = threadIdx.x + 1; }

int main() {
  int runtime_version;
  CHECK_CUDA(cudaRuntimeGetVersion(&runtime_version));
  check("CUDA runtime of the headers", runtime_version == CUDART_VERSION);

  // x = 1, 2, ..., n, filled by a kernel.
  float *dx;
  CHECK_CUDA(cudaMalloc(&dx, n * sizeof(float)));
  iota<<<1, n>>>(dx);
  CHECK_CUDA(cudaGetLastError());
  float x[n];
  CHECK_CUDA(cudaMemcpy(x, dx, sizeof(x), cudaMemcpyDeviceToHost));
  bool filled = true;
  for (int i = 0; i < n; ++i)
    filled = filled && x[i] == i + 1;
  check("kernel", filled);

  // x . x = 1 + 4 + ... + n^2.
  cublasHandle_t cublas;
  CHECK_STATUS(cublasCreate(&cublas), CUBLAS_STATUS_SUCCESS);
  float dot;
  CHECK_STATUS(cublasSdot(cublas, n, dx, 1, dx, 1, &dot),
               CUBLAS_STATUS_SUCCESS);
  check_close("cuBLAS", dot, n * (n + 1) * (2 * n + 1) / 6, 0);
  CHECK_STATUS(cublasDestroy(cublas), CUBLAS_STATUS_SUCCESS);

  // The spectrum of x starts with the sum of its elements.
  cufftHandle plan;
  cufftComplex *dspectrum;
  CHECK_CUDA(cudaMalloc(&dspectrum, (n / 2 + 1) * sizeof(cufftComplex)));
  CHECK_STATUS(cufftPlan1d(&plan, n, CUFFT_R2C, 1), CUFFT_SUCCESS);
  CHECK_STATUS(cufftExecR2C(plan, dx, dspectrum), CUFFT_SUCCESS);
  cufftComplex first;
  CHECK_CUDA(cudaMemcpy(&first, dspectrum, sizeof(first),
                        cudaMemcpyDeviceToHost));
  check_close("cuFFT", first.x, n * (n + 1) / 2, 1e-6);
  CHECK_STATUS(cufftDestroy(plan), CUFFT_SUCCESS);
  CHECK_CUDA(cudaFree(dspectrum));

  // Uniform numbers lie in (0, 1].
  curandGenerator_t generator;
  CHECK_STATUS(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT),
               CURAND_STATUS_SUCCESS);
  CHECK_STATUS(curandGenerateUniform(generator, dx, n), CURAND_STATUS_SUCCESS);
  CHECK_CUDA(cudaMemcpy(x, dx, sizeof(x), cudaMemcpyDeviceToHost));
  bool in_range = true;
  for (int i = 0; i < n; ++i)
    in_range = in_range && x[i] > 0 && x[i] <= 1;
  check("cuRAND", in_range);
  CHECK_STATUS(curandDestroyGenerator(generator), CURAND_STATUS_SUCCESS);

  // The Cholesky factor of [[4, 2], [2, 10]] is [[2, 0], [1, 3]].
  double a[4] = {4, 2, 2, 10}, factor[4];
  double *da, *dwork;
  int *dinfo, lwork;
  cusolverDnHandle_t cusolver;
  CHECK_STATUS(cusolverDnCreate(&cusolver), CUSOLVER_STATUS_SUCCESS);
  CHECK_CUDA(cudaMalloc(&da, sizeof(a)));
  CHECK_CUDA(cudaMalloc(&dinfo, sizeof(int)));
  CHECK_CUDA(cudaMemcpy(da, a, sizeof(a), cudaMemcpyHostToDevice));
  CHECK_STATUS(cusolverDnDpotrf_bufferSize(cusolver, CUBLAS_FILL_MODE_LOWER, 2,
                                           da, 2, &lwork),
               CUSOLVER_STATUS_SUCCESS);
  CHECK_CUDA(cudaMalloc(&dwork, lwork * sizeof(double)));
  CHECK_STATUS(cusolverDnDpotrf(cusolver, CUBLAS_FILL_MODE_LOWER, 2, da, 2,
                                dwork, lwork, dinfo),
               CUSOLVER_STATUS_SUCCESS);
  CHECK_CUDA(cudaMemcpy(factor, da, sizeof(factor), cudaMemcpyDeviceToHost));
  check("cuSOLVER", close_to(factor[0], 2, 1e-12) &&
                        close_to(factor[1], 1, 1e-12) &&
                        close_to(factor[3], 3, 1e-12));
  CHECK_STATUS(cusolverDnDestroy(cusolver), CUSOLVER_STATUS_SUCCESS);
  CHECK_CUDA(cudaFree(da));
  CHECK_CUDA(cudaFree(dwork));
  CHECK_CUDA(cudaFree(dinfo));

  // The sparse matrix [[0, 2], [3, 0]] times (1, 1) is (2, 3).
  int offsets[3] = {0, 1, 2}, columns[2] = {1, 0};
  float values[2] = {2, 3}, ones[2] = {1, 1}, y[2];
  int *doffsets, *dcolumns;
  float *dvalues, *dones, *dy;
  CHECK_CUDA(cudaMalloc(&doffsets, sizeof(offsets)));
  CHECK_CUDA(cudaMalloc(&dcolumns, sizeof(columns)));
  CHECK_CUDA(cudaMalloc(&dvalues, sizeof(values)));
  CHECK_CUDA(cudaMalloc(&dones, sizeof(ones)));
  CHECK_CUDA(cudaMalloc(&dy, sizeof(y)));
  CHECK_CUDA(cudaMemcpy(doffsets, offsets, sizeof(offsets),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dcolumns, columns, sizeof(columns),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dvalues, values, sizeof(values),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dones, ones, sizeof(ones), cudaMemcpyHostToDevice));
  cusparseHandle_t cusparse;
  cusparseSpMatDescr_t matrix;
  cusparseDnVecDescr_t vx, vy;
  CHECK_STATUS(cusparseCreate(&cusparse), CUSPARSE_STATUS_SUCCESS);
  CHECK_STATUS(cusparseCreateCsr(&matrix, 2, 2, 2, doffsets, dcolumns, dvalues,
                                 CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                 CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F),
               CUSPARSE_STATUS_SUCCESS);
  CHECK_STATUS(cusparseCreateDnVec(&vx, 2, dones, CUDA_R_32F),
               CUSPARSE_STATUS_SUCCESS);
  CHECK_STATUS(cusparseCreateDnVec(&vy, 2, dy, CUDA_R_32F),
               CUSPARSE_STATUS_SUCCESS);
  const float alpha = 1, beta = 0;
  size_t buffer_size;
  CHECK_STATUS(cusparseSpMV_bufferSize(cusparse,
                                       CUSPARSE_OPERATION_NON_TRANSPOSE,
                                       &alpha, matrix, vx, &beta, vy,
                                       CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT,
                                       &buffer_size),
               CUSPARSE_STATUS_SUCCESS);
  void *dbuffer;
  CHECK_CUDA(cudaMalloc(&dbuffer, buffer_size));
  CHECK_STATUS(cusparseSpMV(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                            matrix, vx, &beta, vy, CUDA_R_32F,
                            CUSPARSE_SPMV_ALG_DEFAULT, dbuffer),
               CUSPARSE_STATUS_SUCCESS);
  CHECK_CUDA(cudaMemcpy(y, dy, sizeof(y), cudaMemcpyDeviceToHost));
  check("cuSPARSE", y[0] == 2 && y[1] == 3);
  CHECK_STATUS(cusparseDestroyDnVec(vx), CUSPARSE_STATUS_SUCCESS);
  CHECK_STATUS(cusparseDestroyDnVec(vy), CUSPARSE_STATUS_SUCCESS);
  CHECK_STATUS(cusparseDestroySpMat(matrix), CUSPARSE_STATUS_SUCCESS);
  CHECK_STATUS(cusparseDestroy(cusparse), CUSPARSE_STATUS_SUCCESS);
  CHECK_CUDA(cudaFree(doffsets));
  CHECK_CUDA(cudaFree(dcolumns));
  CHECK_CUDA(cudaFree(dvalues));
  CHECK_CUDA(cudaFree(dones));
  CHECK_CUDA(cudaFree(dy));
  CHECK_CUDA(cudaFree(dbuffer));

  CHECK_CUDA(cudaFree(dx));
  return test_result();
}
