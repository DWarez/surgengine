#include "kernel.h"
#include <cstdio>
#include <cuda_runtime.h>

__global__ void hello_cuda() { printf("Hello from CUDA kernel!\n"); }

void launch_kernel() {
  hello_cuda<<<1, 1>>>();
  cudaDeviceSynchronize();
}