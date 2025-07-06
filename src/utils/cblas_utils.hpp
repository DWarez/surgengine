#pragma once

#include <cblas.h>
#include <type_traits>

template <typename T>
void cblas_gemm_dispatch(CBLAS_LAYOUT order, CBLAS_TRANSPOSE transa,
                         CBLAS_TRANSPOSE transb, int m, int n, int k, T alpha,
                         const T *A, int lda, const T *B, int ldb, T beta, T *C,
                         int ldc) {
  if constexpr (std::is_same_v<T, float>) {
    cblas_sgemm(order, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C,
                ldc);
  } else if constexpr (std::is_same_v<T, double>) {
    cblas_dgemm(order, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C,
                ldc);
  }
}

template <typename T>
void cblas_axpy_dispatch(int n, T alpha, const T *x, int incx, T *y, int incy) {
  if constexpr (std::is_same_v<T, float>) {
    cblas_saxpy(n, alpha, x, incx, y, incy);
  } else if constexpr (std::is_same_v<T, double>) {
    cblas_daxpy(n, alpha, x, incx, y, incy);
  }
}
