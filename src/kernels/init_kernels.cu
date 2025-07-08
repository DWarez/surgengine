#include <chrono>
#include <core/tensor.cuh>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <kernels/init_kernels.hpp>

using namespace surgengine;

template <typename T>
__global__ void uniform_kernel(T *data, int n, T low, T high,
                               unsigned long long seed) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < n) {
    curandState state;
    curand_init(seed, idx, 0, &state);

    if constexpr (std::is_same_v<T, float>)
      data[idx] = low + (high - low) * curand_uniform(&state);
    else if constexpr (std::is_same_v<T, double>) {
      data[idx] = low + (high - low) * curand_uniform_double(&state);
    }
    // Todo: add half?
    // Todo: graceful exit with global flag because it looks like that assert(0)
    // is a bad practice ¯\_(ツ)_/¯
  }
};

template <typename T>
__global__ void normal_kernel(T *data, int n, T mean, T std,
                              unsigned long long seed) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < n) {
    curandState state;
    curand_init(seed, idx, 0, &state);

    if constexpr (std::is_same_v<T, float>)
      data[idx] = mean + std * curand_normal(&state);
    else if constexpr (std::is_same_v<T, double>) {
      data[idx] = mean + std * curand_normal_double(&state);
    }
    // Todo: add half?
    // Todo: graceful exit with global flag because it looks like that assert(0)
    // is a bad practice ¯\_(ツ)_/¯
  }
};

inline void get_launch_config(int n, int &grid_size, int &block_size) {
  block_size = 256;
  grid_size = (n + block_size - 1) / block_size;
};

inline unsigned long long generate_seed() {
  auto now = std::chrono::high_resolution_clock::now();
  auto duration = now.time_since_epoch();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
};

template <typename T> void uniform_cuda_(Tensor<T> &tensor, T low, T high) {
  static_assert(std::is_floating_point_v<T>,
                "CUDA uniform_ only supports floating point types");

  cudaSetDevice(tensor.device().rank);

  int n = tensor.numel();
  int grid_size, block_size;
  get_launch_config(n, grid_size, block_size);

  unsigned long long seed = generate_seed();
  uniform_kernel<<<grid_size, block_size>>>(tensor.data(), n, low, high, seed);
  cudaDeviceSynchronize();
};

template <typename T> void normal_cuda_(Tensor<T> &tensor, T mean, T std) {
  static_assert(std::is_floating_point_v<T>,
                "CUDA normal_ only supports floating point types");

  cudaSetDevice(tensor.device().rank);

  int n = tensor.numel();
  int grid_size, block_size;
  get_launch_config(n, grid_size, block_size);

  unsigned long long seed = generate_seed();
  normal_kernel<<<grid_size, block_size>>>(tensor.data(), n, mean, std, seed);
  cudaDeviceSynchronize();
}

template void uniform_cuda_<float>(Tensor<float> &tensor, float low,
                                   float high);
template void uniform_cuda_<double>(Tensor<double> &tensor, double low,
                                    double high);

template void normal_cuda_<float>(Tensor<float> &tensor, float mean, float std);
template void normal_cuda_<double>(Tensor<double> &tensor, double mean,
                                   double std);