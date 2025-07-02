#pragma once

#include <core/tensor.cuh>
#include <cstddef>
#include <random>
#include <stdexcept>
#include <vector>

namespace surgengine {
namespace nn {
namespace init {
template <typename T>
void uniform_(Tensor<T> &tensor, T low = -1.0, T high = 1.0) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<T> dis(low, high);

  std::vector<T> values(tensor.numel());

  for (size_t i = 0; i < values.size(); ++i) {
    values[i] = dis(gen);
  }

  if (tensor.device().is_cuda()) {
    cudaSetDevice(tensor.device().rank);
    cudaMemcpy(tensor.data(), values.data(), values.size() * sizeof(T),
               cudaMemcpyHostToDevice);
  } else {
    std::copy(values.begin(), values.end(), tensor.data());
  }
}

template <typename T>
void normal_(Tensor<T> &tensor, T mean = 0.0, T std = 1.0) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<T> dis(mean, std);

  std::vector<T> values(tensor.numel());
  for (size_t i = 0; i < values.size(); ++i) {
    values[i] = dis(gen);
  }

  if (tensor.device().is_cuda()) {
    cudaSetDevice(tensor.device().rank);
    cudaMemcpy(tensor.data(), values.data(), values.size() * sizeof(T),
               cudaMemcpyHostToDevice);
  } else {
    std::copy(values.begin(), values.end(), tensor.data());
  }
}

template <typename T> void xavier_uniform_(Tensor<T> &tensor) {
  if (tensor.shape().size() < 2) {
    throw std::runtime_error(
        "Xavier initialization requires at leasta a 2D tensor");
  }

  int fan_in = tensor.shape()[1];
  int fan_out = tensor.shape()[0];
  T bound = std::sqrt(6.0 / (fan_in + fan_out));
  uniform_(tensor, -bound, bound);
}

template <typename T> void zeros_(Tensor<T> &tensor) { tensor.zero_(); }

template <typename T> void ones_(Tensor<T> &tensor) { tensor.fill_(T(1.0)); }

template <typename T> void constant_(Tensor<T> &tensor, T value) {
  tensor.fill_(value);
}
} // namespace init
} // namespace nn
} // namespace surgengine