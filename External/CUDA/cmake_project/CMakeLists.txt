# A CUDA project as users write them, with CMake's CUDA language and its
# FindCUDAToolkit module, built with Clang as the CUDA compiler (see
# create_cuda_cmake_project_test in External/CUDA/CMakeLists.txt).

cmake_minimum_required(VERSION 3.20)
project(cuda-cmake-project LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

add_executable(project project.cu)
# For cuda_library_test.h.
target_include_directories(project PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(project PRIVATE
  CUDA::cudart
  CUDA::cublas
  CUDA::cufft
  CUDA::curand
  CUDA::cusolver
  CUDA::cusparse
)
