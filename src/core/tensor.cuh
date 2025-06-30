#pragma once

#include <core/storage.cuh>
#include <cstddef>
#include <cuda_fp16.h>
#include <iomanip>
#include <iostream>
#include <kernels/fill_kernel.hpp>
#include <memory>
#include <sstream>
#include <string>
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

  // PRINTING
  std::string get_dtype_name() const {
    if constexpr (std::is_same_v<T, float>)
      return "float32";
    else if constexpr (std::is_same_v<T, double>)
      return "float64";
    else if constexpr (std::is_same_v<T, half>)
      return "float16";
    else if constexpr (std::is_same_v<T, int32_t>)
      return "int32";
    else if constexpr (std::is_same_v<T, int8_t>)
      return "int8";
    else
      return "unknown";
  }

  void print_tensor_data(std::ostream &os, const std::vector<T> &data) const {
    const int max_elements = 6;

    if (ndim() == 1) {
      print_1d(os, data, max_elements);
    } else if (ndim() == 2) {
      print_2d(os, data, max_elements);
    } else {
      // Higher dimensions - show flattened
      os << "tensor([";
      size_t total = numel();
      if (total <= max_elements) {
        for (size_t i = 0; i < total; ++i) {
          os << format_value(data[i]);
          if (i < total - 1)
            os << ", ";
        }
      } else {
        int half = max_elements / 2;
        for (int i = 0; i < half; ++i) {
          os << format_value(data[i]) << ", ";
        }
        os << "..., ";
        for (size_t i = total - half; i < total; ++i) {
          os << format_value(data[i]);
          if (i < total - 1)
            os << ", ";
        }
      }
      os << "])";
    }
  }

  void print_1d(std::ostream &os, const std::vector<T> &data,
                int max_elements) const {
    os << "[";
    int total_elements = shape_[0];

    if (total_elements <= max_elements) {
      // Print all elements
      for (int i = 0; i < total_elements; ++i) {
        os << format_value(data[i]);
        if (i < total_elements - 1)
          os << ", ";
      }
    } else {
      // Print first few, ..., last few
      int half = max_elements / 2;
      for (int i = 0; i < half; ++i) {
        os << format_value(data[i]) << ", ";
      }
      os << "..., ";
      for (int i = total_elements - half; i < total_elements; ++i) {
        os << format_value(data[i]);
        if (i < total_elements - 1)
          os << ", ";
      }
    }
    os << "]";
  }

  void print_2d(std::ostream &os, const std::vector<T> &data,
                int max_elements) const {
    int rows = shape_[0];
    int cols = shape_[1];

    os << "[" << std::endl;

    if (rows <= max_elements) {
      // Print all rows
      for (int r = 0; r < rows; ++r) {
        os << "  [";
        print_row(os, data, r, cols, max_elements);
        os << "]";
        if (r < rows - 1)
          os << ",";
        os << std::endl;
      }
    } else {
      // Print first few rows, ..., last few rows
      int half = max_elements / 2;
      for (int r = 0; r < half; ++r) {
        os << "  [";
        print_row(os, data, r, cols, max_elements);
        os << "]," << std::endl;
      }
      os << "  ...," << std::endl;
      for (int r = rows - half; r < rows; ++r) {
        os << "  [";
        print_row(os, data, r, cols, max_elements);
        os << "]";
        if (r < rows - 1)
          os << ",";
        os << std::endl;
      }
    }

    os << "]";
  }

  void print_row(std::ostream &os, const std::vector<T> &data, int row,
                 int cols, int max_elements) const {
    if (cols <= max_elements) {
      // Print all columns
      for (int c = 0; c < cols; ++c) {
        os << format_value(data[row * cols + c]);
        if (c < cols - 1)
          os << ", ";
      }
    } else {
      // Print first few, ..., last few columns
      int half = max_elements / 2;
      for (int c = 0; c < half; ++c) {
        os << format_value(data[row * cols + c]) << ", ";
      }
      os << "..., ";
      for (int c = cols - half; c < cols; ++c) {
        os << format_value(data[row * cols + c]);
        if (c < cols - 1)
          os << ", ";
      }
    }
  }

  void print_nd(std::ostream &os, const std::vector<T> &data,
                int max_elements) const {
    // For higher dimensions, show shape and a flattened view
    os << "tensor([";

    size_t total = numel();
    if (total <= max_elements) {
      for (size_t i = 0; i < total; ++i) {
        os << format_value(data[i]);
        if (i < total - 1)
          os << ", ";
      }
    } else {
      int half = max_elements / 2;
      for (int i = 0; i < half; ++i) {
        os << format_value(data[i]) << ", ";
      }
      os << "..., ";
      for (size_t i = total - half; i < total; ++i) {
        os << format_value(data[i]);
        if (i < total - 1)
          os << ", ";
      }
    }
    os << "])";
  }

  std::string format_value(const T &value) const {
    std::ostringstream oss;
    if constexpr (std::is_floating_point_v<T>) {
      oss << std::fixed << std::setprecision(4) << value;
    } else {
      oss << value;
    }
    return oss.str();
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

  // PRINTING
  friend std::ostream &operator<<(std::ostream &os, const Tensor<T> &tensor) {
    os << "Tensor(";
    os << "shape=[";
    for (size_t i = 0; i < tensor.shape_.size(); ++i) {
      os << tensor.shape_[i];
      if (i < tensor.shape_.size() - 1)
        os << ", ";
    }
    os << "], ";
    os << "device="
       << (tensor.device().is_cuda()
               ? "cuda:" + std::to_string(tensor.device().rank)
               : "cpu");
    os << ", dtype=" << tensor.get_dtype_name();
    os << ", numel=" << tensor.numel();
    os << ")" << std::endl;

    if (tensor.numel() == 0) {
      os << "[]";
      return os;
    }

    // Get data on CPU for printing
    std::vector<T> print_data;
    if (tensor.is_cuda()) {
      print_data.resize(tensor.numel());
      cudaMemcpy(print_data.data(), tensor.data(), tensor.numel() * sizeof(T),
                 cudaMemcpyDeviceToHost);
    } else {
      const T *cpu_data = tensor.data();
      print_data.assign(cpu_data, cpu_data + tensor.numel());
    }

    tensor.print_tensor_data(os, print_data);
    return os;
  }

  const std::vector<int> &shape() const { return shape_; }
  const std::vector<int> &strides() const { return strides_; }
  int ndim() const { return shape_.size(); }
  size_t numel() const { return total_elements(); }
  Device device() const {
    return storage_ ? storage_->device() : Device::cpu();
  }

  T *data() { return storage_ ? static_cast<T *>(storage_->data()) : nullptr; }

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

  void fill_(T value) {
    if (!storage_)
      return;

    if (is_cuda()) {
      launch_fill_kernel(data(), value, numel(), device().rank);
    } else {
      auto *cpu_storage = static_cast<CPUStorage<T> *>(storage_.get());
      std::fill(cpu_storage->vector().begin(), cpu_storage->vector().end(),
                value);
    }
  }

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

  static Tensor ones(const std::vector<int> &shape,
                     const Device &device = Device::cpu()) {
    Tensor tensor(shape, device);
    tensor.fill_(T{1});
    return tensor;
  }
};

using FloatTensor = Tensor<float>;
using HalfTensor = Tensor<half>;
using IntTensor = Tensor<int32_t>;
using Int8Tensor = Tensor<int8_t>;
} // namespace surgengine