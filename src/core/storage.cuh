#pragma once

#include <core/device.hpp>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
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

  CUDAStorage(const CUDAStorage &) = delete;
  CUDAStorage &operator=(const CUDAStorage &) = delete;

  void *data() override { return data_; }
  const void *data() const override { return data_; }
  size_t size_bytes() const override { return count_ * sizeof(T); }
  Device device() const override { return Device::cuda(device_rank_); }

  std::unique_ptr<Storage> clone() const override {
    auto new_storage = std::make_unique<CUDAStorage<T>>(count_, device_rank_);
    cudaSetDevice(device_rank_);
    cudaMemcpy(new_storage->data_, data_, count_ * sizeof(T),
               cudaMemcpyDeviceToDevice);
    return std::move(new_storage);
  }

  std::unique_ptr<Storage>
  to_device(const Device &target_device) const override;
};

template <typename T>
std::unique_ptr<Storage> make_storage(size_t count, const Device &device) {
  if (device.is_cuda()) {
    return std::make_unique<CUDAStorage<T>>(count, device.rank);
  }
  return std::make_unique<CPUStorage<T>>(count);
}

template <typename T>
std::unique_ptr<Storage>
CPUStorage<T>::to_device(const Device &target_device) const {
  if (target_device.is_cpu()) {
    return clone();
  }

  auto cuda_storage = std::make_unique<CUDAStorage<T>>(data_.size());
  cudaSetDevice(target_device.rank);
  cudaMemcpy(cuda_storage->data(), data_.data(), size_bytes(),
             cudaMemcpyHostToDevice);
  return cuda_storage;
}

template <typename T>
std::unique_ptr<Storage>
CUDAStorage<T>::to_device(const Device &target_device) const {
  if (target_device.is_cuda() && target_device.rank == device_rank_) {
    return clone();
  } else if (target_device.is_cpu()) {
    auto cpu_storage = std::make_unique<CPUStorage<T>>(count_);
    cudaSetDevice(device_rank_);
    cudaMemcpy(cpu_storage->data(), data_, size_bytes(),
               cudaMemcpyDeviceToHost);
    return cpu_storage;
  } else {
    // CUDA to different CUDA device
    auto cuda_storage =
        std::make_unique<CUDAStorage<T>>(count_, target_device.rank);
    cudaMemcpyPeer(cuda_storage->data(), target_device.rank, data_,
                   device_rank_, size_bytes());
    return cuda_storage;
  }
}
} // namespace surgengine
