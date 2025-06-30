#pragma once

#include <cstddef>
template <typename T>
void launch_fill_kernel(T *data, T value, size_t numel, int device_rank);