#include <cstdio>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <utils/cuda_utils.cuh>

template <typename T> __global__ void fill_kernel(T *data, T value, size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  for (size_t i = idx; i < n; i += stride) {
    data[i] = value;
  }
}

template <typename T>
void launch_fill_kernel(T *data, T value, size_t numel, int device_rank) {
  cudaSetDevice(device_rank);

  const int block_size = 256;
  const int grid_size = (numel + block_size - 1) / block_size;

  // Todo: this limitation can be avoided using streams, but I will not do it
  // right now
  const int max_grid_size = 65535;
  const int final_grid_size = std::min(grid_size, max_grid_size);

  fill_kernel<<<final_grid_size, block_size>>>(data, value, numel);

  CUDA_CHECK(cudaGetLastError());
  cudaDeviceSynchronize();
}

template void launch_fill_kernel<float>(float *, float, size_t, int);
template void launch_fill_kernel<double>(double *, double, size_t, int);
template void launch_fill_kernel<int>(int *, int, size_t, int);
template void launch_fill_kernel<long>(long *, long, size_t, int);
template void launch_fill_kernel<bool>(bool *, bool, size_t, int);