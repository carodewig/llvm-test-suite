//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Every header of the CUDA libraries and every device header of the toolkit
// which the toolkit has, compiled together in every test variant, as Clang
// failing to compile a new toolkit's headers is how new toolkits most often
// break it. A kernel then uses some of the inline device functions of the
// device headers, which Clang compiles, and checks their results.
//
// The headers of the CUDA C++ library, libcu++, are only included from C++17
// on, which the toolkits' newer versions of it require, and so are those which
// include it from CUDA 13 on, such as cooperative_groups.h.

#include "cuda_library_test.h"

#if __has_include(<cublas_v2.h>)
#include <cublas_v2.h>
#endif
#if __has_include(<cublasLt.h>)
#include <cublasLt.h>
#endif
#if __has_include(<cublasXt.h>)
#include <cublasXt.h>
#endif
#if __has_include(<cufft.h>)
#include <cufft.h>
#endif
#if __has_include(<cufftXt.h>)
#include <cufftXt.h>
#endif
#if __has_include(<curand.h>)
#include <curand.h>
#endif
#if __has_include(<curand_kernel.h>)
#include <curand_kernel.h>
#define HAVE_CURAND_KERNEL 1
#endif
#if __has_include(<cusolverDn.h>)
#include <cusolverDn.h>
#endif
#if __has_include(<cusolverSp.h>)
#include <cusolverSp.h>
#endif
#if __has_include(<cusolverRf.h>)
#include <cusolverRf.h>
#endif
#if __has_include(<cusolverMg.h>)
#include <cusolverMg.h>
#endif
#if __has_include(<cusparse.h>)
#include <cusparse.h>
#endif
#if __has_include(<npp.h>)
#include <npp.h>
#endif
#if __has_include(<nvjpeg.h>)
#include <nvjpeg.h>
#endif
#if __has_include(<nvrtc.h>)
#include <nvrtc.h>
#endif
#if __has_include(<nvJitLink.h>)
#include <nvJitLink.h>
#endif
#if __has_include(<cuda_fp16.h>)
#include <cuda_fp16.h>
#define HAVE_FP16 1
#endif
#if __has_include(<cuda_bf16.h>)
#include <cuda_bf16.h>
#define HAVE_BF16 1
#endif
#if __has_include(<cuda_fp8.h>)
#include <cuda_fp8.h>
#endif
// Whether the C++ standard is one which libcu++ supports, where the toolkit's
// headers include it.
#define LIBCUXX_STANDARD (CUDART_VERSION < 13000 || __cplusplus >= 201703L)
#if __has_include(<cooperative_groups.h>) && LIBCUXX_STANDARD
#include <cooperative_groups.h>
#define HAVE_COOPERATIVE_GROUPS 1
#endif
#if __has_include(<cooperative_groups/reduce.h>) && LIBCUXX_STANDARD &&       \
    __cplusplus >= 201103L
#include <cooperative_groups/reduce.h>
#endif
// libcu++ gains headers from one toolkit to the next.
#if __cplusplus >= 201703L
#if __has_include(<cuda/std/algorithm>)
#include <cuda/std/algorithm>
#endif
#if __has_include(<cuda/std/atomic>)
#include <cuda/std/atomic>
#endif
#if __has_include(<cuda/std/array>) && __has_include(<cuda/std/cstdint>) &&    \
    __has_include(<cuda/std/limits>)
#include <cuda/std/array>
#include <cuda/std/cstdint>
#include <cuda/std/limits>
#define HAVE_LIBCUXX 1
#endif
#endif

const int threads = 64;

// Each thread computes the same results with the device headers' functions
// where the toolkit has them, and with plain arithmetic where it does not, and
// records whether they are right.
__global__ void use_headers(int *wrong) {
  __shared__ int sum;
  int failed = 0;

#if HAVE_FP16 && __CUDA_ARCH__ >= 530
  __half h = __hadd(__float2half(1.5f), __float2half(2.25f));
  failed |= __half2float(h) != 3.75f;
#endif
#if HAVE_BF16 && __CUDA_ARCH__ >= 800
  __nv_bfloat16 b = __hmul(__float2bfloat16(2.0f), __float2bfloat16(3.0f));
  failed |= __bfloat162float(b) != 6.0f;
#endif
#if HAVE_CURAND_KERNEL
  curandStatePhilox4_32_10_t state;
  curand_init(0, threadIdx.x, 0, &state);
  float u = curand_uniform(&state);
  failed |= !(u > 0 && u <= 1);
#endif
#if HAVE_LIBCUXX
  cuda::std::array<int, 3> array = {{1, 2, 3}};
  failed |= array.size() != 3 || array[2] != 3;
  failed |= cuda::std::numeric_limits<cuda::std::int32_t>::max() != 2147483647;
#endif

  if (threadIdx.x == 0)
    sum = 0;
#if HAVE_COOPERATIVE_GROUPS
  cooperative_groups::thread_block block =
      cooperative_groups::this_thread_block();
  block.sync();
  atomicAdd(&sum, (int)threadIdx.x);
  block.sync();
#else
  __syncthreads();
  atomicAdd(&sum, (int)threadIdx.x);
  __syncthreads();
#endif
  failed |= sum != threads * (threads - 1) / 2;

  if (failed)
    atomicAdd(wrong, 1);
}

int main() {
  int *dwrong;
  CHECK_CUDA(cudaMalloc(&dwrong, sizeof(int)));
  CHECK_CUDA(cudaMemset(dwrong, 0, sizeof(int)));
  use_headers<<<1, threads>>>(dwrong);
  CHECK_CUDA(cudaGetLastError());
  int wrong;
  CHECK_CUDA(cudaMemcpy(&wrong, dwrong, sizeof(int), cudaMemcpyDeviceToHost));
  check("device functions of the headers", wrong == 0);
  CHECK_CUDA(cudaFree(dwrong));
  return test_result();
}
