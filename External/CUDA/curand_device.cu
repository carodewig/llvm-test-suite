//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cuda_runtime.h>
#include <stdio.h>

// The test is verified with CUDA 12 and newer.
#if CUDA_VERSION < 12000

int main() {
  printf("Success!\n");
  return 0;
}

#else

// cuRAND's device API is implemented in curand_kernel.h and compiled by Clang.
#include <curand_kernel.h>

// Philox4x32-10 turns a four-word counter and a two-word key into four output
// words. These input/output pairs from Random123's tests/kat_vectors check
// cuRAND's implementation bit for bit.
struct KnownAnswer {
  unsigned counter[4];
  unsigned key[2];
  unsigned result[4];
};

const KnownAnswer known_answers[] = {
    {{0, 0, 0, 0}, {0, 0},
     {0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8}},
    {{0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff},
     {0xffffffff, 0xffffffff},
     {0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd}},
    {{0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344},
     {0xa4093822, 0x299f31d0},
     {0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1}},
};
const int num_known_answers = sizeof(known_answers) / sizeof(known_answers[0]);

__global__ void philox(const KnownAnswer *answers, uint4 *results) {
  int i = threadIdx.x;
  const KnownAnswer &a = answers[i];
  uint4 counter = make_uint4(a.counter[0], a.counter[1], a.counter[2],
                             a.counter[3]);
  uint2 key = make_uint2(a.key[0], a.key[1]);
  results[i] = curand_Philox4x32_10(counter, key);
}

// With seed, subsequence and offset zero, curand_init starts at counter and
// key zero. The first four curand calls should return the four words of
// known_answers[0].result, also checking cuRAND's state-based interface.
__global__ void first_numbers(unsigned *numbers) {
  curandStatePhilox4_32_10_t state;
  curand_init(0, 0, 0, &state);
  for (int i = 0; i < 4; ++i)
    numbers[i] = curand(&state);
}

static bool check(cudaError_t err) {
  if (err != cudaSuccess)
    printf("CUDA error %d: %s\n", (int)err, cudaGetErrorString(err));
  return err == cudaSuccess;
}

static bool test_known_answers() {
  KnownAnswer *danswers;
  uint4 *dresults;
  if (!check(cudaMalloc(&danswers, sizeof(known_answers))) ||
      !check(cudaMalloc(&dresults, num_known_answers * sizeof(uint4))) ||
      !check(cudaMemcpy(danswers, known_answers, sizeof(known_answers),
                        cudaMemcpyHostToDevice)))
    return false;
  philox<<<1, num_known_answers>>>(danswers, dresults);
  uint4 results[num_known_answers];
  if (!check(cudaGetLastError()) ||
      !check(cudaMemcpy(results, dresults, sizeof(results),
                        cudaMemcpyDeviceToHost)))
    return false;

  for (int i = 0; i < num_known_answers; ++i) {
    const unsigned *expected = known_answers[i].result;
    uint4 got = results[i];
    if (got.x != expected[0] || got.y != expected[1] ||
        got.z != expected[2] || got.w != expected[3]) {
      printf("Philox known answer %d differs\n", i);
      return false;
    }
  }
  return check(cudaFree(danswers)) && check(cudaFree(dresults));
}

static bool test_first_numbers() {
  unsigned *dnumbers;
  if (!check(cudaMalloc(&dnumbers, 4 * sizeof(unsigned))))
    return false;
  first_numbers<<<1, 1>>>(dnumbers);
  unsigned numbers[4];
  if (!check(cudaGetLastError()) ||
      !check(cudaMemcpy(numbers, dnumbers, sizeof(numbers),
                        cudaMemcpyDeviceToHost)))
    return false;

  for (int i = 0; i < 4; ++i) {
    if (numbers[i] != known_answers[0].result[i]) {
      printf("Philox first number %d differs\n", i);
      return false;
    }
  }
  return check(cudaFree(dnumbers));
}

int main() {
  if (!test_known_answers() || !test_first_numbers())
    return 1;
  printf("Success!\n");
  return 0;
}

#endif
