#pragma once

#include <cstddef>
#include <cstdio>
#include <utils/cuda_utils.cuh>
#include <vector>

namespace surgengine {
enum class DataType {
  FLOAT32,
  FLOAT16,
  INT32,
  INT8,
};

enum class Device {
  CPU,
  CUDA,
};

template <typename T> class CudaPtr {
private:
  T *ptr_;
  size_t size_;

public:
  CudaPtr() : ptr_(nullptr), size_(0) {}

  explicit CudaPtr(size_t count) : size_(count) {
    CUDA_CHECK(cudaMalloc(&ptr_, count * sizeof(T)));
  }

  ~CudaPtr() {
    if (ptr_) {
      cudaFree(ptr_);
    }
  }

  CudaPtr(CudaPtr &&other) noexcept : ptr_(other.ptr_), size_(other.size_) {
    other.size_ = 0;
    other.ptr_ = nullptr;
  }

  CudaPtr &operator=(CudaPtr &&other) noexcept {
    if (this != &other) {
      if (ptr_)
        cudaFree(ptr_);
      ptr_ = other.ptr_;
      size_ = other.size_;
      other.size_ = 0;
      other.ptr_ = nullptr;
    }
    return *this;
  }

  CudaPtr(const CudaPtr &) = delete;
  CudaPtr &operator=(const CudaPtr &) = delete;

  T *get() const { return ptr_; }
  size_t size() const { return size_; }

  void reset(size_t count = 0) {
    if (ptr_) {
      cudaFree(ptr_);
      ptr_ = nullptr;
      size_ = 0;
    }
    if (count > 0) {
      CUDA_CHECK(cudaMalloc(&ptr_, count * sizeof(T)));
      size_ = count;
    }
  }
};

template <typename T> class Tensor {
private:
  std::vector<int> shape_;
  std::vector<int> strides;
  CudaPtr<T> cuda_data_;
};
} // namespace surgengine