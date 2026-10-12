//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A cuFFT two-dimensional complex-to-complex transform of an impulse, whose
// spectrum is known in closed form: an impulse at (p, q) transforms to
// exp(-2 pi i (k p / nx + l q / ny)) at (k, l). Transforming it back gives the
// impulse scaled by the size of the transform. The two dimensions differ, so
// mixing them up fails.

#include <cufft.h>

#include "cuda_library_test.h"

#define CHECK_CUFFT(call) CHECK_STATUS(call, CUFFT_SUCCESS)

// Row-major: element (i, j) is at i * ny + j.
const int nx = 8;
const int ny = 6;
const int size = nx * ny;
// Where the impulse is.
const int p = 3;
const int q = 1;

int main() {
  const double pi = acos(-1.0);
  cufftComplex x[size];
  float impulse[2 * size], expected[2 * size];
  for (int i = 0; i < size; ++i) {
    x[i].x = x[i].y = 0;
    impulse[2 * i] = impulse[2 * i + 1] = 0;
  }
  x[p * ny + q].x = 1;
  impulse[2 * (p * ny + q)] = 1;
  for (int k = 0; k < nx; ++k) {
    for (int l = 0; l < ny; ++l) {
      double phase = -2 * pi * ((double)k * p / nx + (double)l * q / ny);
      expected[2 * (k * ny + l)] = cos(phase);
      expected[2 * (k * ny + l) + 1] = sin(phase);
    }
  }

  cufftComplex *dx;
  CHECK_CUDA(cudaMalloc(&dx, sizeof(x)));
  CHECK_CUDA(cudaMemcpy(dx, x, sizeof(x), cudaMemcpyHostToDevice));
  cufftHandle plan;
  CHECK_CUFFT(cufftPlan2d(&plan, nx, ny, CUFFT_C2C));

  CHECK_CUFFT(cufftExecC2C(plan, dx, dx, CUFFT_FORWARD));
  CHECK_CUDA(cudaMemcpy(x, dx, sizeof(x), cudaMemcpyDeviceToHost));
  check_all_close("cufftExecC2C 2D spectrum", (float *)x, expected, 2 * size,
                  1e-5);

  CHECK_CUFFT(cufftExecC2C(plan, dx, dx, CUFFT_INVERSE));
  CHECK_CUDA(cudaMemcpy(x, dx, sizeof(x), cudaMemcpyDeviceToHost));
  for (int i = 0; i < size; ++i) {
    x[i].x /= size;
    x[i].y /= size;
  }
  check_all_close("cufftExecC2C 2D back to the impulse", (float *)x, impulse,
                  2 * size, 1e-5);

  CHECK_CUFFT(cufftDestroy(plan));
  CHECK_CUDA(cudaFree(dx));
  return test_result();
}
