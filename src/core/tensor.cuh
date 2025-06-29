#pragma once

#include <core/storage.cuh>
#include <cstddef>
#include <cuda_fp16.h>
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

  const T *data() {
    return storage_ ? static_cast<const T *>(storage_->data()) : nullptr;
  }

  const T *data() const {
    return storage_ ? static_cast<const T *>(storage_->data()) : nullptr;
  }

  Tensor to(const Device &target_device) const {
    if (!storage_ || storage_->device() == target_device)
      return *this;

    Tensor moved_tensor;
    moved_tensor.shape_ = shape_;
    moved_tensor.strides_ = strides_;
    moved_tensor.storage_ = storage_->to_device(target_device);
    return moved_tensor;
  }

  Tensor &to_(const Device &target_device) {
    if (storage_ && storage_->device() != target_device) {
      storage_ = storage_->to_device(target_device);
    }
    return *this;
  }

  Tensor cuda(int device_rank = 0) const {
    return to(Device::cuda(device_rank));
  }

  Tensor cpu() const { return to(Device::cpu()); }

  Tensor &cuda_(int device_rank = 0) { return to_(Device::cuda(device_rank)); }

  Tensor &cpu_() { return to_(Device::cpu()); }

  bool is_cuda() const { return device().is_cuda(); }
  bool is_cpu() const { return device().is_cpu(); }

  Tensor view(const std::vector<int> &new_shape) const {
    size_t new_total = 1;
    for (int dim : new_shape) {
      new_total *= dim;
    }

    if (new_total != total_elements()) {
      throw std::invalid_argument(
          "Cannot reshape tensor: element count mismatch");
    }

    Tensor result = *this;
    result.shape_ = new_shape;
    result.compute_strides();
    return result;
  }

  // Todo: to implement filling kernel
  // void fill_(T value) {
  //   if (!storage_)
  //     return;

  //   if (is_cuda()) {
  //     launch_fill_kernel(data(), value, numel(), device().rank);
  //   } else {
  //     auto *cpu_storage = static_cast<CPUStorage<T> *>(storage_.get());
  //     std::fill(cpu_storage->vector().begin(), cpu_storage->vector().end(),
  //               value);
  //   }
  // }

  void zero_() {
    if (!storage_)
      return;

    if (is_cuda()) {
      cudaSetDevice(device().rank);
      cudaMemset(data(), 0, storage_->size_bytes());
    } else {
      auto *cpu_storage = static_cast<CPUStorage<T> *>(storage_.get());
      std::fill(cpu_storage->vector().begin(), cpu_storage->vector().end(),
                T{0});
    }
  }

  static Tensor zeros(const std::vector<int> &shape,
                      const Device &device = Device::cpu()) {
    Tensor tensor(shape, device);
    tensor.zero_();
    return tensor;
  }

  // Todo: enable when fill method is complete
  // static Tensor ones(const std::vector<int> &shape,
  //                    const Device &device = Device::cpu()) {
  //   Tensor tensor(shape, device);
  //   tensor.fill_(T{1});
  //   return tensor;
  // }
};

using FloatTensor = Tensor<float>;
using HalfTensor = Tensor<half>;
using IntTensor = Tensor<int32_t>;
using Int8Tensor = Tensor<int8_t>;
} // namespace surgengine