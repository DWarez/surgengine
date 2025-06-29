#pragma once

#include <core/storage.cuh>
#include <cstddef>
#include <memory>
#include <utils/cuda_utils.cuh>
#include <vector>

namespace surgengine {
enum class DataType {
  FLOAT32,
  FLOAT16,
  INT32,
  INT8,
};

template <typename T> class Tensor {
private:
  std::vector<int> shape_;
  std::vector<int> strides_;
  std::unique_ptr<Storage> storage_;

  void compute_strides() {
    strides_.resize(shape_.size());

    if (shape_.empty())
      return;

    strides_.back() = 1;

    for (int i = shape_.size() - 2; i >= 0; --i) {
      strides_[i] = strides_[i + 1] * shape_[i + 1];
    }
  }

  size_t total_elements() const {
    size_t total = 1;
    for (int dim : shape_) {
      total *= dim;
    }
    return total;
  }

public:
  Tensor() = default;

  Tensor(const std::vector<int> &shape, const Device &device = Device::cpu())
      : shape_(shape) {
    compute_strides();
    storage_ = surgengine::make_storage<T>(total_elements(), device);
  }

  Tensor(const Tensor &other) : shape_(other.shape_), strides_(other.strides_) {
    if (other.storage_)
      storage_ = other.storage_->clone();
  }

  Tensor(Tensor &&other) noexcept
      : shape_(std::move(other.shape_)), strides_(std::move(other.strides_)),
        storage_(std::move(other.storage_)) {}

  Tensor &operator=(const Tensor &other) {
    if (this != &other) {
      shape_ = other.shape_;
      strides_ = other.strides_;
      storage_ = other.storage_ ? other.storage_->clone() : nullptr;
    }
    return this;
  }

  Tensor &operator=(Tensor &&other) noexcept {
    if (this != &other) {
      shape_ = std::move(other.shape_);
      strides_ = std::move(other.strides_);
      storage_ = std::move(other.storage_);
    }
    return *this;
  }

  const std::vector<int> &shape() const { return shape_; }
  const std::vector<int> &strides() const { return strides_; }
  int ndim() const { return shape_.size(); }
  size_t numel() const { return total_elements(); }
  Device device() const {
    return storage_ ? storage_->device() : Device::cpu();
  }
};
} // namespace surgengine