#pragma once

#include <core/device.hpp>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <exception>
#include <memory>
#include <stdexcept>
#include <utils/cuda_utils.cuh>
#include <vector>

using Device = surgengine::Device;

namespace surgengine {
// Abstract class for StorageTypes
class Storage {
public:
  virtual ~Storage() = default;
  virtual void *data() = 0;
  virtual const void *data() const = 0;
  virtual size_t size_bytes() const = 0;
  virtual Device device() const = 0;
  virtual std::unique_ptr<Storage> clone() const = 0;
  virtual std::unique_ptr<Storage> to_device(const Device &device) const = 0;
};

// CPU Class
template <typename T> class CPUStorage : public Storage {
private:
  std::vector<T> data_;

public:
  explicit CPUStorage(size_t count) : data_(count) {}

  CPUStorage(const CPUStorage &&other) noexcept
      : data_(std::move(other.data_)) {}

  CPUStorage &operator=(CPUStorage &other) noexcept {
    if (this != &other) {
      data_ = std::move(other.data_);
    }
    return this;
  }

  CPUStorage(const CPUStorage &) = delete;
  CPUStorage &operator=(const CPUStorage &) = delete;

  void *data() override { return data_.data(); }
  const void *data() const override { return data_.data(); }
  size_t size_bytes() const override { return data_.size() * sizeof(T); }
  Device device() const override { return Device::cpu(); }

  std::unique_ptr<Storage> clone() const override {
    auto new_storage = std::make_unique<CPUStorage<T>>(data_.size());
    std::copy(data_.begin(), data_.end(), new_storage->data_.begin());
    return new_storage;
  }

  std::unique_ptr<Storage>
  to_device(const Device &target_device) const override;

  std::vector<T> &vector() { return data_; }
  const std::vector<T> &vector() const { return data_; }
};

template <typename T> class CUDAStorage : public Storage {
private:
  T *data_;
  size_t count_;
  int device_rank_;

public:
  explicit CUDAStorage(size_t count, int device_rank = 0)
      : count_(count), device_rank_(device_rank) {
    cudaSetDevice(device_rank_);
    CUDA_CHECK(cudaMalloc(&data_, count * sizeof(T)));
  }

  ~CUDAStorage() {
    if (data_) {
      cudaSetDevice(device_rank_);
      cudaFree(data_);
    }
  }

  CUDAStorage(CUDAStorage &&other) noexcept
      : data_(other.data_), count_(other.count_),
        device_rank_(other.device_rank_) {
    other.data_ = nullptr;
    other.count_ = 0;
  }

  CUDAStorage &operator=(CUDAStorage &&other) {
    if (this != &other) {
      if (data_) {
        cudaSetDevice(device_rank_);
        cudaFree(data_);
        data_ = nullptr;
        count_ = 0;
      }
      if (device_rank_ == other.device_rank_) {
        data_ = other.data_;
        count_ = other.count_;
        other.data_ = nullptr;
        other.count_ = 0;
      } else {
        throw std::runtime_error(
            "Trying to move a CUDAStorage into another that has a different "
            "GPU rank. This is not allowed since it would require implicit "
            "data movements.");
      }
    }
    return *this;
  }
};
} // namespace surgengine
