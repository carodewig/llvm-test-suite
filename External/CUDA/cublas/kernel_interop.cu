//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Kernels compiled by Clang and cuBLAS working on the same data, in order on
// one stream: kernels fill the operands, cuBLAS multiplies them, and another
// kernel checks the product on the device. The host checks the product too.

#include <cublas_v2.h>

#include "cuda_library_test.h"

#define CHECK_CUBLAS(call) CHECK_STATUS(call, CUBLAS_STATUS_SUCCESS)

// A is m x k, B is k x n and C is m x n, all column-major.
const int m = 32;
const int n = 24;
const int k = 16;
const int threads = 128;

__host__ __device__ float element(int i, int mul, int mod) {
  return (i * mul) % mod - mod / 2;
}

__global__ void fill(float *x, int size, int mul, int mod) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size)
    x[i] = element(i, mul, mod);
}

// Count the elements of C which are not the product of A and B.
__global__ void verify(const float *a, const float *b, const float *c,
                       int *mismatches) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= m * n)
    return;
  int i = index % m, j = index / m;
  float sum = 0;
  for (int l = 0; l < k; ++l)
    sum += a[l * m + i] * b[j * k + l];
  if (c[index] != sum)
    atomicAdd(mismatches, 1);
}

static int blocks(int size) { return (size + threads - 1) / threads; }

int main() {
  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));
  cublasHandle_t handle;
  CHECK_CUBLAS(cublasCreate(&handle));
  CHECK_CUBLAS(cublasSetStream(handle, stream));

  float *da, *db, *dc;
  int *dmismatches;
  CHECK_CUDA(cudaMalloc(&da, m * k * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&db, k * n * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dc, m * n * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dmismatches, sizeof(int)));

  // Nothing waits for the stream until the end, so every step must follow the
  // one before it on the stream.
  CHECK_CUDA(cudaMemsetAsync(dmismatches, 0, sizeof(int), stream));
  fill<<<blocks(m * k), threads, 0, stream>>>(da, m * k, 5, 7);
  CHECK_CUDA(cudaGetLastError());
  fill<<<blocks(k * n), threads, 0, stream>>>(db, k * n, 3, 5);
  CHECK_CUDA(cudaGetLastError());
  const float alpha = 1, beta = 0;
  CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha,
                           da, m, db, k, &beta, dc, m));
  verify<<<blocks(m * n), threads, 0, stream>>>(da, db, dc, dmismatches);
  CHECK_CUDA(cudaGetLastError());

  float c[m * n];
  int mismatches;
  CHECK_CUDA(cudaMemcpyAsync(c, dc, sizeof(c), cudaMemcpyDeviceToHost, stream));
  CHECK_CUDA(cudaMemcpyAsync(&mismatches, dmismatches, sizeof(int),
                             cudaMemcpyDeviceToHost, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));

  check("product checked on the device", mismatches == 0);

  float expected[m * n];
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      float sum = 0;
      for (int l = 0; l < k; ++l)
        sum += element(l * m + i, 5, 7) * element(j * k + l, 3, 5);
      expected[j * m + i] = sum;
    }
  }
  check_all_close("product checked on the host", c, expected, m * n, 0);

  CHECK_CUBLAS(cublasDestroy(handle));
  CHECK_CUDA(cudaStreamDestroy(stream));
  CHECK_CUDA(cudaFree(da));
  CHECK_CUDA(cudaFree(db));
  CHECK_CUDA(cudaFree(dc));
  CHECK_CUDA(cudaFree(dmismatches));
  return test_result();
}
