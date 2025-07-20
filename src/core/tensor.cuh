#pragma once

#include "utils/cuda_utils.cuh"
#include <atomic>
#include <core/device.cuh>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <kernels/fill_kernel.hpp>
#include <memory>
#include <new>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace surgengine {
template <typename T> class Allocator {
public:
  static T *allocate(size_t count, size_t alignment = 32) {
    size_t size = count * sizeof(T);

    if (alignment < sizeof(void *))
      alignment = sizeof(void *);

    // bit manipulation magic to round to the closest power of 2
    size = (size + alignment - 1) & ~(alignment - 1);
    void *ptr = std::aligned_alloc(alignment, size);

    if (!ptr) {
      throw std::bad_alloc();
    }
    return static_cast<T *>(ptr);
  }

  static void deallocate(T *ptr) { std::free(ptr); }
};

template <typename T> class RefCountedStorage {
private:
  T *data_;
  size_t size_;
  size_t capacity_;
  mutable std::atomic<int> ref_count_;
  Device device_;

public:
  RefCountedStorage(size_t size, const Device &device)
      : size_(size), capacity_(size), ref_count_(1), device_(device) {
    if (device.is_cpu()) {
      data_ = Allocator<T>::allocate(capacity_, 32);

    } else {
      cudaSetDevice(device_.rank);
      CUDA_CHECK(cudaMalloc(&data_, capacity_ * sizeof(T)));
    }
  }

  ~RefCountedStorage() {
    if (device_.is_cpu())
      Allocator<T>::deallocate(data_);
    else {
      cudaSetDevice(device_.rank);
      cudaFree(data_);
    }
  }

  void addref() const { ref_count_.fetch_add(1); }
  void release() const {
    if (ref_count_.fetch_sub(1) == 1) {
      delete this;
    }
  }

  T *data() const { return data_; }
  T *data() { return data_; }
  size_t size() const { return size_; };
  Device device() const { return device_; };

  RefCountedStorage(const RefCountedStorage &) = delete;
  RefCountedStorage &operator=(const RefCountedStorage &) = delete;
};

template <typename T> using StoragePtr = std::shared_ptr<RefCountedStorage<T>>;

class TensorShape {
  std::vector<int> dims_;
  std::vector<int> strides_;

private:
  void compute_strides() {
    strides_.resize(dims_.size());
    if (dims_.empty())
      return;

    strides_.back() = 1;
    for (int i = dims_.size() - 2; i >= 0; --i) {
      strides_[i] = strides_[i + 1] * dims_[i + 1];
    }
  }

public:
  TensorShape(const std::vector<int> &shape) : dims_(shape) {
    compute_strides();
  }

  TensorShape(const std::vector<int> &shape, const std::vector<int> &strides)
      : dims_(shape), strides_(strides) {}

  const std::vector<int> &dims() const { return dims_; }
  const std::vector<int> &strides() const { return strides_; }

  size_t numel() const {
    size_t total = 1;
    for (int dim : dims_)
      total *= dim;
    return total;
  }
};

template <typename T> class Tensor {
private:
  StoragePtr<T> storage_;
  TensorShape shape_;
  size_t offset_;

  static void copy_data_between_devices(const float *src, float *dst,
                                        size_t count, const Device &src_device,
                                        const Device &dst_device) {

    // this case should probably never happen but let's support it because why
    // not
    if (src_device.is_cpu() && dst_device.is_cpu()) {
      std::memcpy(dst, src, count * sizeof(float));
    } else if (src_device.is_cpu() && dst_device.is_cuda()) {
      cudaSetDevice(dst_device.rank);
      CUDA_CHECK(
          cudaMemcpy(dst, src, count * sizeof(float), cudaMemcpyHostToDevice));
    } else if (src_device.is_cuda() && dst_device.is_cpu()) {
      cudaSetDevice(src_device.rank);
      CUDA_CHECK(
          cudaMemcpy(dst, src, count * sizeof(float), cudaMemcpyDeviceToHost));
    } else if (src_device.is_cuda() && dst_device.is_cuda()) {
      if (src_device.rank == dst_device.rank) {
        cudaSetDevice(src_device.rank);
        CUDA_CHECK(cudaMemcpy(dst, src, count * sizeof(float),
                              cudaMemcpyDeviceToDevice));
      } else {
        CUDA_CHECK(cudaMemcpyPeer(dst, dst_device.rank, src, src_device.rank,
                                  count * sizeof(float)));
      }
    }
  }

public:
  Tensor(const std::vector<int> &shape, const Device &device = Device::cpu())
      : shape_(shape), offset_(0) {
    size_t total = shape_.numel();
    storage_ = std::make_shared<RefCountedStorage<T>>(total, device);
  }

  Tensor(const TensorShape &shape, const Device &device = Device::cpu())
      : shape_(shape), offset_(0) {
    size_t total = shape_.numel();
    storage_ = std::make_shared<RefCountedStorage<T>>(total, device);
  }

  Tensor(StoragePtr<T> storage, const std::vector<int> &shape,
         const std::vector<int> &strides, size_t offset = 0)
      : storage_(storage), shape_(shape, strides), offset_(offset) {}

  T *data() { return storage_ ? storage_->data() + offset_ : nullptr; }
  const T *data() const {
    return storage_ ? storage_->data() + offset_ : nullptr;
  }

  const std::vector<int> &shape() const { return shape_.dims(); }
  const std::vector<int> &strides() const { return shape_.strides(); }
  size_t numel() const { return shape_.numel(); }
  Device device() const {
    return storage_ ? storage_->device() : Device::cpu();
  }

  bool is_contiguous() const {
    std::vector<int> expected_strides = shape_.dims();
    if (expected_strides.empty())
      return true;

    expected_strides.back() = 1;
    for (int i = expected_strides.size() - 2; i >= 0; --i) {
      expected_strides[i] = expected_strides[i + 1] * shape_.dims()[i + 1];
    }
    return shape_.strides() == expected_strides;
  }

  Tensor &zero_() {
    if (!storage_)
      return *this;

    if (device().is_cuda()) {
      cudaSetDevice(device().rank);
      cudaMemset(data(), 0, numel() * sizeof(T));
    } else {
      memset(data(), 0, numel() * sizeof(T));
    }
    return *this;
  }

  Tensor &fill_(T value) {
    if (!storage_)
      return *this;

    if (device().is_cuda()) {
      launch_fill_kernel(data(), value, numel(), device().rank);
    } else {
      std::fill_n(data(), numel(), value);
    }
    return *this;
  }

  Tensor view(const std::vector<int> &new_shape) const {
    if (!is_contiguous()) {
      throw std::runtime_error("Cannot reshape non-contiguous tensor");
    }

    size_t new_total = 1;
    for (int dim : new_shape)
      new_total *= dim;

    if (new_total != numel()) {
      throw std::invalid_argument("Cannot reshape: element count mismatch");
    }

    std::vector<int> new_strides(new_shape.size());
    if (!new_shape.empty()) {
      new_strides.back() = 1;
      for (int i = new_shape.size() - 2; i >= 0; --i) {
        new_strides[i] = new_strides[i + 1] * new_shape[i + 1];
      }
    }

    return Tensor(storage_, new_shape, new_strides, offset_);
  }

  Tensor slice(size_t dim, int start, int end) const {
    if (dim >= shape_.dims().size()) {
      throw std::out_of_range("Dimension out of range");
    }

    if (start < 0 || end > shape_.dims()[dim] || start >= end) {
      throw std::out_of_range("Slice indices out of range");
    }

    std::vector<int> new_shape = shape_.dims();
    new_shape[dim] = end - start;

    size_t new_offset = offset_ + start * shape_.strides()[dim];

    return Tensor(storage_, new_shape, shape_.strides(), new_offset);
  }

  static Tensor zeros(const std::vector<int> &shape,
                      const Device &device = Device::cpu()) {
    Tensor tensor(shape, device);
    tensor.zero_();
    return tensor;
  }

  static Tensor empty(const std::vector<int> &shape,
                      const Device &device = Device::cpu()) {
    return Tensor(shape, device);
  }

  Tensor to(const Device &target_device) const {
    if (device() == target_device)
      return *this;

    Tensor<T> result(shape_, target_device);
    copy_data_between_devices(storage_->data(), result.storage_->data(),
                              numel(), storage_->device(), target_device);
    return result;
  }

  Tensor cuda(int rank = 0) const { return to(Device::cuda(rank)); }

  Tensor cpu() const { return to(Device(Device::Device::cpu())); }

  size_t memory_usage() const {
    return storage_ ? storage_->size() * sizeof(T) : 0;
  }

  bool shares_storage(const Tensor &other) const {
    return storage_ == other.storage_;
  }

  // PRINT METHODS
private:
  std::string get_dtype_name() const {
    if constexpr (std::is_same_v<T, float>) {
      return "float32";
    } else if constexpr (std::is_same_v<T, double>) {
      return "float64";
    } else if constexpr (std::is_same_v<T, int32_t>) {
      return "int32";
    } else if constexpr (std::is_same_v<T, int8_t>) {
      return "int8";
    } else if constexpr (std::is_same_v<T, int16_t>) {
      return "int16";
    } else if constexpr (std::is_same_v<T, int64_t>) {
      return "int64";
    } else {
      return "unknown";
    }
  }

  void print_tensor_data(std::ostream &os, const std::vector<T> &data) const {
    const int max_elements = 6;

    if (shape_.dims().size() == 0) {
      os << format_value(data[0]);
    } else if (shape_.dims().size() == 1) {
      print_1d(os, data, max_elements);
    } else if (shape_.dims().size() == 2) {
      print_2d(os, data, max_elements);
    } else {
      print_nd(os, data, max_elements);
    }
  }

  void print_1d(std::ostream &os, const std::vector<T> &data,
                int max_elements) const {
    os << "[";
    int total_elements = shape_.dims()[0];

    if (total_elements <= max_elements) {
      for (int i = 0; i < total_elements; ++i) {
        os << format_value(data[i]);
        if (i < total_elements - 1)
          os << ", ";
      }
    } else {
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
    int rows = shape_.dims()[0];
    int cols = shape_.dims()[1];

    os << "[" << std::endl;

    if (rows <= max_elements) {
      for (int r = 0; r < rows; ++r) {
        os << "  [";
        print_row(os, data, r, cols, max_elements);
        os << "]";
        if (r < rows - 1)
          os << ",";
        os << std::endl;
      }
    } else {
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
      for (int c = 0; c < cols; ++c) {
        os << format_value(data[row * cols + c]);
        if (c < cols - 1)
          os << ", ";
      }
    } else {
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
    os << "tensor([";

    size_t total = numel();
    if (total <= (size_t)(max_elements)) {
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
  friend std::ostream &operator<<(std::ostream &os, const Tensor<T> &tensor) {
    os << "Tensor(";
    os << "shape=[";
    for (size_t i = 0; i < tensor.shape_.dims().size(); ++i) {
      os << tensor.shape_.dims()[i];
      if (i < tensor.shape_.dims().size() - 1)
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
    std::vector<T> print_data;
    if (tensor.device().is_cuda()) {
      print_data.resize(tensor.numel());
      cudaSetDevice(tensor.device().rank);
      cudaMemcpy(print_data.data(), tensor.data(), tensor.numel() * sizeof(T),
                 cudaMemcpyDeviceToHost);
    } else {
      const T *cpu_data = tensor.data();
      print_data.assign(cpu_data, cpu_data + tensor.numel());
    }

    tensor.print_tensor_data(os, print_data);
    return os;
  }
};

using FloatTensor = Tensor<float>;
using DoubleTensor = Tensor<double>;
} // namespace surgengine
