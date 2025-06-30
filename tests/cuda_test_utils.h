#pragma once

#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace test_utils {
#define CUDA_CHECK_TEST(call)                                                  \
  do {                                                                         \
    cudaError_t error = call;                                                  \
    if (error != cudaSuccess) {                                                \
      throw std::runtime_error(std::string("CUDA error: ") +                   \
                               cudaGetErrorString(error) + " at " + __FILE__ + \
                               ":" + std::to_string(__LINE__));                \
    }                                                                          \
  } while (0)

// Check if CUDA is available and skip test if not
#define REQUIRE_CUDA()                                                         \
  do {                                                                         \
    int device_count = 0;                                                      \
    cudaGetDeviceCount(&device_count);                                         \
    if (device_count == 0) {                                                   \
      GTEST_SKIP() << "CUDA not available";                                    \
    }                                                                          \
  } while (0)

// Require specific number of CUDA devices
#define REQUIRE_CUDA_DEVICES(n)                                                \
  do {                                                                         \
    int device_count = 0;                                                      \
    cudaGetDeviceCount(&device_count);                                         \
    if (device_count < n) {                                                    \
      GTEST_SKIP() << "Test requires " << n << " CUDA devices, only "          \
                   << device_count << " available";                            \
    }                                                                          \
  } while (0)

template <typename T>
void compare_gpu_cpu_data(const T *gpu_data, const std::vector<T> &cpu_data,
                          size_t count, T tolerance = T{1e-6}) {
  std::vector<T> gpu_host_data(count);
  CUDA_CHECK_TEST(cudaMemcpy(gpu_host_data.data(), gpu_data, count * sizeof(T),
                             cudaMemcpyHostToDevice));
  for (size_t i = 0; i < count; ++i) {
    if constexpr (std::is_floating_point_v<T>) {
      EXPECT_NEAR(gpu_host_data[i], cpu_data[i], tolerance)
          << "Missmatch at index " << i;
    } else {
      EXPECT_EQ(gpu_host_data[i], cpu_data[i]) << "Missmatch at index " << i;
    }
  }
}

template <typename T> std::vector<T> create_test_pattern(size_t count) {
  std::vector<T> data(count);
  for (size_t i = 0; i < count; ++i) {
    if constexpr (std::is_floating_point_v<T>) {
      data[i] = static_cast<T>(std::sin(i * 0.1) * 100.0);
    } else {
      data[i] = static_cast<T>(i % 127);
    }
  }
  return data;
}

class MemoryChecker {
private:
  size_t initial_free_memory_;
  size_t initial_total_memory_;

public:
  MemoryChecker() {
    if (get_cuda_device_count() > 0) {
      cudaMemGetInfo(&initial_free_memory_, &initial_total_memory_);
    }
  }

  ~MemoryChecker() {
    if (get_cuda_device_count() > 0) {
      size_t current_free, current_total;
      cudaMemGetInfo(&current_free, &current_total);

      // Allow some tolerance for CUDA driver overhead
      const size_t tolerance = 1024 * 1024; // 1MB

      if (current_free + tolerance < initial_free_memory_) {
        ADD_FAILURE() << "Potential memory leak detected. "
                      << "Initial free: " << initial_free_memory_ / 1024 / 1024
                      << "MB, "
                      << "Current free: " << current_free / 1024 / 1024 << "MB";
      }
    }
  }

private:
  int get_cuda_device_count() {
    int count = 0;
    cudaGetDeviceCount(&count);
    return count;
  }
};

class Timer {
private:
  cudaEvent_t start_, stop_;
  bool use_cuda_;

public:
  explicit Timer(bool use_cuda = false) : use_cuda_(use_cuda) {
    if (use_cuda_) {
      CUDA_CHECK_TEST(cudaEventCreate(&start_));
      CUDA_CHECK_TEST(cudaEventCreate(&stop_));
    }
  }

  ~Timer() {
    if (use_cuda_) {
      cudaEventDestroy(start_);
      cudaEventDestroy(stop_);
    }
  }

  void start() {
    if (use_cuda_) {
      CUDA_CHECK_TEST(cudaEventRecord(start_));
    }
  }

  float stop() {
    if (use_cuda_) {
      CUDA_CHECK_TEST(cudaEventRecord(stop_));
      CUDA_CHECK_TEST(cudaEventSynchronize(stop_));

      float milliseconds = 0;
      CUDA_CHECK_TEST(cudaEventElapsedTime(&milliseconds, start_, stop_));
      return milliseconds;
    }
    return 0.0f;
  }
};

class CUDATestWithMemoryCheck : public ::testing::Test {
protected:
  void SetUp() override {
    REQUIRE_CUDA();
    memory_checker_ = std::make_unique<MemoryChecker>();
    cudaSetDevice(0);
  }

  void TearDown() override {
    cudaDeviceSynchronize();
    memory_checker_.reset(); // Check for leaks
    cudaDeviceReset();
  }

private:
  std::unique_ptr<MemoryChecker> memory_checker_;
};
} // namespace test_utils

#define EXPECT_TENSOR_EQ(tensor1, tensor2)                                     \
  do {                                                                         \
    EXPECT_EQ((tensor1).shape(), (tensor2).shape());                           \
    EXPECT_EQ((tensor1).device().type, (tensor2).device().type);               \
    if ((tensor1).is_cpu() && (tensor2).is_cpu()) {                            \
      auto *data1 = (tensor1).data();                                          \
      auto *data2 = (tensor2).data();                                          \
      for (size_t i = 0; i < (tensor1).numel(); ++i) {                         \
        EXPECT_EQ(data1[i], data2[i]) << "Tensors differ at index " << i;      \
      }                                                                        \
    }                                                                          \
  } while (0)

#define EXPECT_TENSOR_NEAR(tensor1, tensor2, tolerance)                        \
  do {                                                                         \
    EXPECT_EQ((tensor1).shape(), (tensor2).shape());                           \
    if ((tensor1).is_cpu() && (tensor2).is_cpu()) {                            \
      auto *data1 = (tensor1).data();                                          \
      auto *data2 = (tensor2).data();                                          \
      for (size_t i = 0; i < (tensor1).numel(); ++i) {                         \
        EXPECT_NEAR(data1[i], data2[i], tolerance)                             \
            << "Tensors differ at index " << i;                                \
      }                                                                        \
    }                                                                          \
  } while (0)