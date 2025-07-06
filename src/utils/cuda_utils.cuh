#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

template <typename T>
void cublas_gemm_dispatch(cublasHandle_t handle, cublasOperation_t transa,
                          cublasOperation_t transb, int m, int n, int k,
                          const T *alpha, const T *A, int lda, const T *B,
                          int ldb, const T *beta, T *C, int ldc) {
  cublasStatus_t status = CUBLAS_STATUS_NOT_SUPPORTED;
  if constexpr (std::is_same_v<T, float>) {
    status = cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb,
                         beta, C, ldc);
  } else if constexpr (std::is_same_v<T, double>) {
    status = cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb,
                         beta, C, ldc);
  } else if constexpr (std::is_same_v<T, __half>) {
    status = cublasHgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb,
                         beta, C, ldc);
  }
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error("cuBLAS GEMM failed with status: " +
                             std::to_string(status));
  }
}

template <typename T>
void cublas_axpy_dispatch(cublasHandle_t cublas_handle, int n, const T *alpha,
                          const T *x, int incx, T *y, int incy) {
  if constexpr (std::is_same_v<T, float>) {
    cublasSaxpy(cublas_handle, n, alpha, x, incx, y, incy);
  } else if constexpr (std::is_same_v<T, double>) {
    cublasDaxpy(cublas_handle, n, alpha, x, incx, y, incy);
  } else if constexpr (std::is_same_v<T, __half>) {
    cublasHaxpy(cublas_handle, n, alpha, x, incx, y, incy);
  }
}

// Basic CUDA error checking macro
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t error = call;                                                  \
    if (error != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__,        \
              cudaGetErrorString(error));                                      \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// Alternative macro with custom error handling
#define CUDA_CHECK_RETURN(call)                                                \
  do {                                                                         \
    cudaError_t error = call;                                                  \
    if (error != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__,        \
              cudaGetErrorString(error));                                      \
      return error;                                                            \
    }                                                                          \
  } while (0)

// Macro for checking kernel launches (requires explicit synchronization)
#define CUDA_CHECK_KERNEL()                                                    \
  do {                                                                         \
    cudaError_t error = cudaGetLastError();                                    \
    if (error != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA kernel launch error at %s:%d - %s\n", __FILE__,    \
              __LINE__, cudaGetErrorString(error));                            \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
    error = cudaDeviceSynchronize();                                           \
    if (error != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA kernel execution error at %s:%d - %s\n", __FILE__, \
              __LINE__, cudaGetErrorString(error));                            \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)
