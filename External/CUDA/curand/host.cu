//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// cuRAND's host API: numbers generated on the device with a fixed seed. They
// must lie in the range of their distribution and have its mean and variance,
// which a million numbers estimate to within much less than the tolerances.
// The same seed must give the same numbers again.

#include <curand.h>

#include "cuda_library_test.h"

#define CHECK_CURAND(call) CHECK_STATUS(call, CURAND_STATUS_SUCCESS)

const int n = 1 << 20;
const unsigned long long seed = 1234;

static void moments(const float *x, double *mean, double *variance) {
  double sum = 0, sum_squares = 0;
  for (int i = 0; i < n; ++i) {
    sum += x[i];
    sum_squares += (double)x[i] * x[i];
  }
  *mean = sum / n;
  *variance = sum_squares / n - *mean * *mean;
}

static void generate(float *dx, bool normal) {
  curandGenerator_t generator;
  CHECK_CURAND(
      curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_PHILOX4_32_10));
  CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(generator, seed));
  if (normal)
    CHECK_CURAND(curandGenerateNormal(generator, dx, n, 0, 1));
  else
    CHECK_CURAND(curandGenerateUniform(generator, dx, n));
  CHECK_CURAND(curandDestroyGenerator(generator));
}

int main() {
  float *x = (float *)malloc(n * sizeof(float));
  float *again = (float *)malloc(n * sizeof(float));
  float *dx;
  CHECK_CUDA(cudaMalloc(&dx, n * sizeof(float)));
  double mean, variance;

  generate(dx, false);
  CHECK_CUDA(cudaMemcpy(x, dx, n * sizeof(float), cudaMemcpyDeviceToHost));
  // curandGenerateUniform excludes 0 and includes 1.
  bool in_range = true;
  for (int i = 0; i < n; ++i)
    in_range = in_range && x[i] > 0 && x[i] <= 1;
  check("curandGenerateUniform range", in_range);
  moments(x, &mean, &variance);
  check_close("curandGenerateUniform mean", mean, 0.5, 0.005);
  check_close("curandGenerateUniform variance", variance, 1.0 / 12, 0.005);

  generate(dx, false);
  CHECK_CUDA(cudaMemcpy(again, dx, n * sizeof(float), cudaMemcpyDeviceToHost));
  bool same = true;
  for (int i = 0; i < n; ++i)
    same = same && x[i] == again[i];
  check("curandGenerateUniform same seed, same numbers", same);

  generate(dx, true);
  CHECK_CUDA(cudaMemcpy(x, dx, n * sizeof(float), cudaMemcpyDeviceToHost));
  moments(x, &mean, &variance);
  check_close("curandGenerateNormal mean", mean, 0, 0.01);
  check_close("curandGenerateNormal variance", variance, 1, 0.01);

  CHECK_CUDA(cudaFree(dx));
  free(x);
  free(again);
  return test_result();
}
