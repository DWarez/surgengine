#include <cuda_runtime.h>

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
