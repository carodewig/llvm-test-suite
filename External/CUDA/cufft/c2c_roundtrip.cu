//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// cuFFT complex-to-complex transforms: a forward transform followed by an
// inverse one gives back the input scaled by its size, as cuFFT does not
// normalize. Sizes which are a power of two and which are not take different
// paths through cuFFT.
//
// The rounding errors scale with the size of the values, so the result is
// scaled back by the size and compared with the input.

#include <cufft.h>

#include "cuda_library_test.h"

#define CHECK_CUFFT(call) CHECK_STATUS(call, CUFFT_SUCCESS)

static void test_roundtrip(const char *name, int n) {
  cufftComplex *x = (cufftComplex *)malloc(n * sizeof(cufftComplex));
  cufftComplex *expected = (cufftComplex *)malloc(n * sizeof(cufftComplex));
  for (int i = 0; i < n; ++i) {
    x[i].x = (i * 5) % 7 - 3;
    x[i].y = (i * 3) % 4 - 1.5f;
    expected[i] = x[i];
  }

  cufftComplex *dx;
  CHECK_CUDA(cudaMalloc(&dx, n * sizeof(cufftComplex)));
  CHECK_CUDA(cudaMemcpy(dx, x, n * sizeof(cufftComplex),
                        cudaMemcpyHostToDevice));
  cufftHandle plan;
  CHECK_CUFFT(cufftPlan1d(&plan, n, CUFFT_C2C, 1));
  CHECK_CUFFT(cufftExecC2C(plan, dx, dx, CUFFT_FORWARD));
  CHECK_CUFFT(cufftExecC2C(plan, dx, dx, CUFFT_INVERSE));
  CHECK_CUDA(cudaMemcpy(x, dx, n * sizeof(cufftComplex),
                        cudaMemcpyDeviceToHost));
  for (int i = 0; i < n; ++i) {
    x[i].x /= n;
    x[i].y /= n;
  }
  // The real and imaginary parts, one after the other.
  check_all_close(name, (float *)x, (float *)expected, 2 * n, 1e-5);

  CHECK_CUFFT(cufftDestroy(plan));
  CHECK_CUDA(cudaFree(dx));
  free(x);
  free(expected);
}

int main() {
  test_roundtrip("cufftExecC2C round trip, size 64", 64);
  test_roundtrip("cufftExecC2C round trip, size 60", 60);
  return test_result();
}
