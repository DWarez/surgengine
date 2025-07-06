#pragma once
#include <cuda_runtime.h>
#include <utils/cuda_utils.cuh>

namespace surgengine {
enum class DeviceType { CPU, CUDA };

struct Device {
  DeviceType type;
  int rank;

  Device(DeviceType t = DeviceType::CPU, int rank = 0) : type(t), rank(rank) {}

  static Device cpu() { return Device(DeviceType::CPU, 0); }
  static Device cuda(int rank = 0) { return Device(DeviceType::CUDA, rank); }

  bool operator==(const Device &other) const {
    return type == other.type && rank == other.rank;
  }

  bool operator!=(const Device &other) const { return !(*this == other); }

  bool is_cuda() const { return type == DeviceType::CUDA; }
  bool is_cpu() const { return type == DeviceType::CPU; }

  static int cuda_device_count() {
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    return device_count;
  }

  static bool is_cuda_available() {
    if (cuda_device_count() > 0)
      return true;
    return false;
  }
};
} // namespace surgengine