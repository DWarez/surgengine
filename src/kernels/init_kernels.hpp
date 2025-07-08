#pragma once

#include <core/tensor.cuh>

using namespace surgengine;

inline void get_launch_config(int n, int &grid_size, int &block_size);

inline unsigned long long generate_seed();

template <typename T>
void uniform_cuda_(Tensor<T> &tensor, T low = -1.0, T high = 1.0);

template <typename T>
void normal_cuda_(Tensor<T> &tensor, T mean = 0.0, T std = 1.0);