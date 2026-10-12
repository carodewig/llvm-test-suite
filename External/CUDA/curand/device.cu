//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// cuRAND's device API, which is code in curand_kernel.h, so Clang compiles it
// into the kernels here, unlike the GPU code of the other libraries.
//
// Its Philox4x32-10 generator must give the known answers of the Random123
// library, where the generator comes from, bit for bit, both called directly
// and as the first numbers of a generator state with seed, subsequence and
// offset 0. Uniform and normal numbers from many states must have the mean and
// variance of their distribution.

#include <curand_kernel.h>

#include "cuda_library_test.h"

// Random123's known answers for Philox4x32-10: counter, key and result.
struct KnownAnswer {
  unsigned counter[4];
  unsigned key[2];
  unsigned result[4];
};
const KnownAnswer known_answers[] = {
    {{0x00000000, 0x00000000, 0x00000000, 0x00000000},
     {0x00000000, 0x00000000},
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

// The first four numbers of a state with seed, subsequence and offset 0, which
// are those of the counter and key 0.
__global__ void first_numbers(unsigned *numbers) {
  curandStatePhilox4_32_10_t state;
  curand_init(0, 0, 0, &state);
  for (int i = 0; i < 4; ++i)
    numbers[i] = curand(&state);
}

// Each thread draws from its own subsequence of the same seed.
const int threads = 256;
const int blocks = 64;
const int per_thread = 64;
const int n = threads * blocks * per_thread;

__global__ void distributions(float *uniform, float *normal) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  curandStatePhilox4_32_10_t state;
  curand_init(1234, id, 0, &state);
  for (int i = 0; i < per_thread; ++i) {
    uniform[id * per_thread + i] = curand_uniform(&state);
    normal[id * per_thread + i] = curand_normal(&state);
  }
}

static void moments(const float *x, double *mean, double *variance) {
  double sum = 0, sum_squares = 0;
  for (int i = 0; i < n; ++i) {
    sum += x[i];
    sum_squares += (double)x[i] * x[i];
  }
  *mean = sum / n;
  *variance = sum_squares / n - *mean * *mean;
}

int main() {
  KnownAnswer *danswers;
  uint4 *dresults;
  CHECK_CUDA(cudaMalloc(&danswers, sizeof(known_answers)));
  CHECK_CUDA(cudaMalloc(&dresults, num_known_answers * sizeof(uint4)));
  CHECK_CUDA(cudaMemcpy(danswers, known_answers, sizeof(known_answers),
                        cudaMemcpyHostToDevice));
  philox<<<1, num_known_answers>>>(danswers, dresults);
  CHECK_CUDA(cudaGetLastError());
  uint4 results[num_known_answers];
  CHECK_CUDA(cudaMemcpy(results, dresults, sizeof(results),
                        cudaMemcpyDeviceToHost));
  bool all_known = true;
  for (int i = 0; i < num_known_answers; ++i) {
    const unsigned *r = known_answers[i].result;
    all_known = all_known && results[i].x == r[0] && results[i].y == r[1] &&
                results[i].z == r[2] && results[i].w == r[3];
  }
  check("curand_Philox4x32_10 known answers", all_known);

  unsigned *dnumbers;
  CHECK_CUDA(cudaMalloc(&dnumbers, 4 * sizeof(unsigned)));
  first_numbers<<<1, 1>>>(dnumbers);
  CHECK_CUDA(cudaGetLastError());
  unsigned numbers[4];
  CHECK_CUDA(cudaMemcpy(numbers, dnumbers, sizeof(numbers),
                        cudaMemcpyDeviceToHost));
  bool first_known = true;
  for (int i = 0; i < 4; ++i)
    first_known = first_known && numbers[i] == known_answers[0].result[i];
  check("curand first numbers of seed 0", first_known);

  float *uniform = (float *)malloc(n * sizeof(float));
  float *normal = (float *)malloc(n * sizeof(float));
  float *duniform, *dnormal;
  CHECK_CUDA(cudaMalloc(&duniform, n * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dnormal, n * sizeof(float)));
  distributions<<<blocks, threads>>>(duniform, dnormal);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaMemcpy(uniform, duniform, n * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(normal, dnormal, n * sizeof(float),
                        cudaMemcpyDeviceToHost));
  // curand_uniform excludes 0 and includes 1.
  bool in_range = true;
  for (int i = 0; i < n; ++i)
    in_range = in_range && uniform[i] > 0 && uniform[i] <= 1;
  check("curand_uniform range", in_range);
  double mean, variance;
  moments(uniform, &mean, &variance);
  check_close("curand_uniform mean", mean, 0.5, 0.005);
  check_close("curand_uniform variance", variance, 1.0 / 12, 0.005);
  moments(normal, &mean, &variance);
  check_close("curand_normal mean", mean, 0, 0.01);
  check_close("curand_normal variance", variance, 1, 0.01);

  CHECK_CUDA(cudaFree(danswers));
  CHECK_CUDA(cudaFree(dresults));
  CHECK_CUDA(cudaFree(dnumbers));
  CHECK_CUDA(cudaFree(duniform));
  CHECK_CUDA(cudaFree(dnormal));
  free(uniform);
  free(normal);
  return test_result();
}
