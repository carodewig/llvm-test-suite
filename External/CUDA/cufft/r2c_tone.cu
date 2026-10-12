//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// cuFFT real-to-complex and complex-to-real transforms of a signal whose
// spectrum is known: a constant plus a cosine at one frequency, which give a
// spike at frequency 0 and one at the cosine's frequency, and nothing else.
// Transforming the spectrum back gives the signal scaled by its size.
//
// The rounding errors scale with the size of the values, so the results are
// scaled by the size before they are compared.

#include <cufft.h>

#include "cuda_library_test.h"

#define CHECK_CUFFT(call) CHECK_STATUS(call, CUFFT_SUCCESS)

const int n = 64;
// The frequency of the cosine.
const int tone = 5;
// A real transform of size n has n / 2 + 1 frequencies, the others being their
// complex conjugates.
const int bins = n / 2 + 1;

int main() {
  const double pi = acos(-1.0);
  float signal[n];
  for (int t = 0; t < n; ++t)
    signal[t] = 1 + cos(2 * pi * tone * t / n);

  // The spectrum, divided by n: the constant 1 gives n at frequency 0, and the
  // cosine n / 2 at its frequency.
  float expected[2 * bins];
  for (int i = 0; i < 2 * bins; ++i)
    expected[i] = 0;
  expected[0] = 1;
  expected[2 * tone] = 0.5f;

  float *dsignal;
  cufftComplex *dspectrum;
  CHECK_CUDA(cudaMalloc(&dsignal, n * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dspectrum, bins * sizeof(cufftComplex)));
  CHECK_CUDA(cudaMemcpy(dsignal, signal, sizeof(signal),
                        cudaMemcpyHostToDevice));

  cufftHandle forward, inverse;
  CHECK_CUFFT(cufftPlan1d(&forward, n, CUFFT_R2C, 1));
  CHECK_CUFFT(cufftPlan1d(&inverse, n, CUFFT_C2R, 1));

  cufftComplex spectrum[bins];
  CHECK_CUFFT(cufftExecR2C(forward, dsignal, dspectrum));
  CHECK_CUDA(cudaMemcpy(spectrum, dspectrum, sizeof(spectrum),
                        cudaMemcpyDeviceToHost));
  float scaled[2 * bins];
  for (int i = 0; i < bins; ++i) {
    scaled[2 * i] = spectrum[i].x / n;
    scaled[2 * i + 1] = spectrum[i].y / n;
  }
  check_all_close("cufftExecR2C spectrum", scaled, expected, 2 * bins, 1e-5);

  float back[n];
  CHECK_CUFFT(cufftExecC2R(inverse, dspectrum, dsignal));
  CHECK_CUDA(cudaMemcpy(back, dsignal, sizeof(back), cudaMemcpyDeviceToHost));
  for (int t = 0; t < n; ++t)
    back[t] /= n;
  check_all_close("cufftExecC2R back to the signal", back, signal, n, 1e-5);

  CHECK_CUFFT(cufftDestroy(forward));
  CHECK_CUFFT(cufftDestroy(inverse));
  CHECK_CUDA(cudaFree(dsignal));
  CHECK_CUDA(cudaFree(dspectrum));
  return test_result();
}
