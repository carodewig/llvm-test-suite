//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Helpers for the tests of the CUDA libraries. They build in every test
// variant, so they stick to C++98.
//
// A test exits with a message as soon as a CUDA or library call fails, and
// otherwise prints a line per check, "<name>: PASS" or "<name>: FAIL ...", so
// that its output only matches its reference output when every check passed.

#ifndef CUDA_LIBRARY_TEST_H
#define CUDA_LIBRARY_TEST_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>

// Exit with a message when a CUDA runtime call fails.
#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t err_ = (call);                                                 \
    if (err_ != cudaSuccess) {                                                 \
      printf("%s:%d: %s failed: %s\n", __FILE__, __LINE__, #call,              \
             cudaGetErrorString(err_));                                        \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// Exit with a message when a library call does not return the status 'ok',
// e.g. CUBLAS_STATUS_SUCCESS.
#define CHECK_STATUS(call, ok)                                                 \
  do {                                                                         \
    int status_ = (int)(call);                                                 \
    if (status_ != (int)(ok)) {                                                \
      printf("%s:%d: %s failed: status %d\n", __FILE__, __LINE__, #call,       \
             status_);                                                         \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

static int num_failures = 0;

// Whether 'got' is within 'tol' of 'expected', relative to the size of
// 'expected', or absolute where it is smaller than 1.
static bool close_to(double got, double expected, double tol) {
  double scale = fabs(expected) > 1.0 ? fabs(expected) : 1.0;
  return fabs(got - expected) <= tol * scale;
}

// Report a check which passed or failed as a whole.
static void check(const char *name, bool ok) {
  if (ok) {
    printf("%s: PASS\n", name);
  } else {
    printf("%s: FAIL\n", name);
    ++num_failures;
  }
}

// Report whether a single value is close to what it should be.
static void check_close(const char *name, double got, double expected,
                        double tol) {
  if (close_to(got, expected, tol)) {
    printf("%s: PASS\n", name);
  } else {
    printf("%s: FAIL: got %g, expected %g\n", name, got, expected);
    ++num_failures;
  }
}

// Report whether 'n' values are each close to what they should be, naming the
// first which is not.
template <typename T>
static void check_all_close(const char *name, const T *got, const T *expected,
                            int n, double tol) {
  for (int i = 0; i < n; ++i) {
    if (!close_to(got[i], expected[i], tol)) {
      printf("%s: FAIL at %d: got %g, expected %g\n", name, i, (double)got[i],
             (double)expected[i]);
      ++num_failures;
      return;
    }
  }
  printf("%s: PASS\n", name);
}

// The exit status of a test: whether any of its checks failed.
static int test_result() { return num_failures == 0 ? 0 : 1; }

#endif // CUDA_LIBRARY_TEST_H
