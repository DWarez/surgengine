#include <algorithm>
#include <core/storage.cuh>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <memory>
#include <numeric>
#include <vector>

using namespace surgengine;

class StorageTest : public ::testing::Test {
protected:
  void SetUp() override {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    cuda_available = device_count > 0;

    if (cuda_available) {
      cudaSetDevice(0);
    }
  }

  void TearDown() override {
    if (cuda_available) {
      cudaDeviceReset();
    }
  }

  bool cuda_available = false;
  static constexpr size_t test_size = 1000;
};

// CPUStorage Tests
class CPUStorageTest : public StorageTest {};

TEST_F(CPUStorageTest, ConstructorInitializesCorrectSize) {
  auto storage = std::make_unique<CPUStorage<float>>(test_size);

  EXPECT_EQ(storage->size_bytes(), test_size * sizeof(float));
  EXPECT_NE(storage->data(), nullptr);
  EXPECT_EQ(storage->device(), Device::cpu());
}

TEST_F(CPUStorageTest, DataAccessWorks) {
  auto storage = std::make_unique<CPUStorage<int>>(test_size);

  int *data_ptr = static_cast<int *>(storage->data());
  EXPECT_NE(data_ptr, nullptr);

  const auto &const_storage = *storage;
  const int *const_data_ptr = static_cast<const int *>(const_storage.data());
  EXPECT_NE(const_data_ptr, nullptr);
  EXPECT_EQ(data_ptr, const_data_ptr);
}

TEST_F(CPUStorageTest, VectorAccessWorks) {
  auto storage = std::make_unique<CPUStorage<float>>(test_size);

  auto &vec = storage->vector();
  std::iota(vec.begin(), vec.end(), 1.0f);

  EXPECT_EQ(vec.size(), test_size);
  EXPECT_EQ(vec[0], 1.0f);
  EXPECT_EQ(vec[test_size - 1], static_cast<float>(test_size));

  const auto &const_vec =
      static_cast<const CPUStorage<float> &>(*storage).vector();
  EXPECT_EQ(const_vec.size(), test_size);
  EXPECT_EQ(const_vec[0], 1.0f);
}

TEST_F(CPUStorageTest, MoveConstructorWorks) {
  auto original = std::make_unique<CPUStorage<double>>(test_size);
  auto &vec = original->vector();
  std::iota(vec.begin(), vec.end(), 1.0);

  CPUStorage<double> moved(std::move(*original));

  EXPECT_EQ(moved.size_bytes(), test_size * sizeof(double));
  EXPECT_EQ(moved.vector().size(), test_size);
  EXPECT_EQ(moved.vector()[0], 1.0);
  EXPECT_EQ(moved.vector()[test_size - 1], static_cast<double>(test_size));
}

TEST_F(CPUStorageTest, CloneWorks) {
  auto original = std::make_unique<CPUStorage<int>>(test_size);
  auto &vec = original->vector();
  std::iota(vec.begin(), vec.end(), 1);

  auto cloned = original->clone();

  EXPECT_NE(cloned.get(), original.get());
  EXPECT_EQ(cloned->size_bytes(), original->size_bytes());
  EXPECT_EQ(cloned->device(), original->device());

  auto cloned_cpu = dynamic_cast<CPUStorage<int> *>(cloned.get());
  ASSERT_NE(cloned_cpu, nullptr);

  const auto &cloned_vec = cloned_cpu->vector();
  EXPECT_EQ(cloned_vec.size(), test_size);
  for (size_t i = 0; i < test_size; ++i) {
    EXPECT_EQ(cloned_vec[i], static_cast<int>(i + 1));
  }
}

TEST_F(CPUStorageTest, ToDeviceCPUReturnsCopy) {
  auto original = std::make_unique<CPUStorage<float>>(test_size);
  auto &vec = original->vector();
  std::iota(vec.begin(), vec.end(), 1.0f);

  auto copied = original->to_device(Device::cpu());

  EXPECT_NE(copied.get(), original.get());
  EXPECT_EQ(copied->device(), Device::cpu());
  EXPECT_EQ(copied->size_bytes(), original->size_bytes());

  auto copied_cpu = dynamic_cast<CPUStorage<float> *>(copied.get());
  ASSERT_NE(copied_cpu, nullptr);

  const auto &copied_vec = copied_cpu->vector();
  for (size_t i = 0; i < test_size; ++i) {
    EXPECT_EQ(copied_vec[i], static_cast<float>(i + 1));
  }
}

// CUDAStorage Tests
class CUDAStorageTest : public StorageTest {};

TEST_F(CUDAStorageTest, ConstructorInitializesCorrectSize) {
  if (!cuda_available) {
    GTEST_SKIP() << "CUDA not available";
  }

  auto storage = std::make_unique<CUDAStorage<float>>(test_size, 0);

  EXPECT_EQ(storage->size_bytes(), test_size * sizeof(float));
  EXPECT_NE(storage->data(), nullptr);
  EXPECT_EQ(storage->device(), Device::cuda(0));
}

TEST_F(CUDAStorageTest, MoveConstructorWorks) {
  if (!cuda_available) {
    GTEST_SKIP() << "CUDA not available";
  }

  auto original = std::make_unique<CUDAStorage<double>>(test_size, 0);
  void *original_ptr = original->data();

  CUDAStorage<double> moved(std::move(*original));

  EXPECT_EQ(moved.data(), original_ptr);
  EXPECT_EQ(moved.size_bytes(), test_size * sizeof(double));
  EXPECT_EQ(moved.device(), Device::cuda(0));
}

TEST_F(CUDAStorageTest, CloneWorks) {
  if (!cuda_available) {
    GTEST_SKIP() << "CUDA not available";
  }

  auto original = std::make_unique<CUDAStorage<int>>(test_size, 0);

  std::vector<int> host_data(test_size);
  std::iota(host_data.begin(), host_data.end(), 1);
  cudaMemcpy(original->data(), host_data.data(), original->size_bytes(),
             cudaMemcpyHostToDevice);

  auto cloned = original->clone();

  EXPECT_NE(cloned.get(), original.get());
  EXPECT_NE(cloned->data(), original->data());
  EXPECT_EQ(cloned->size_bytes(), original->size_bytes());
  EXPECT_EQ(cloned->device(), original->device());

  std::vector<int> cloned_data(test_size);
  cudaMemcpy(cloned_data.data(), cloned->data(), cloned->size_bytes(),
             cudaMemcpyDeviceToHost);

  for (size_t i = 0; i < test_size; ++i) {
    EXPECT_EQ(cloned_data[i], static_cast<int>(i + 1));
  }
}

TEST_F(CUDAStorageTest, ToDeviceCUDAReturnsCopy) {
  if (!cuda_available) {
    GTEST_SKIP() << "CUDA not available";
  }

  auto original = std::make_unique<CUDAStorage<float>>(test_size, 0);

  std::vector<float> host_data(test_size);
  std::iota(host_data.begin(), host_data.end(), 1.0f);
  cudaMemcpy(original->data(), host_data.data(), original->size_bytes(),
             cudaMemcpyHostToDevice);

  auto copied = original->to_device(Device::cuda(0));

  EXPECT_NE(copied.get(), original.get());
  EXPECT_NE(copied->data(), original->data());
  EXPECT_EQ(copied->device(), Device::cuda(0));
  EXPECT_EQ(copied->size_bytes(), original->size_bytes());

  std::vector<float> copied_data(test_size);
  cudaMemcpy(copied_data.data(), copied->data(), copied->size_bytes(),
             cudaMemcpyDeviceToHost);

  for (size_t i = 0; i < test_size; ++i) {
    EXPECT_EQ(copied_data[i], static_cast<float>(i + 1));
  }
}

TEST_F(CUDAStorageTest, ToDeviceCPUWorks) {
  if (!cuda_available) {
    GTEST_SKIP() << "CUDA not available";
  }

  auto original = std::make_unique<CUDAStorage<int>>(test_size, 0);

  std::vector<int> host_data(test_size);
  std::iota(host_data.begin(), host_data.end(), 1);
  cudaMemcpy(original->data(), host_data.data(), original->size_bytes(),
             cudaMemcpyHostToDevice);

  auto cpu_storage = original->to_device(Device::cpu());

  EXPECT_EQ(cpu_storage->device(), Device::cpu());
  EXPECT_EQ(cpu_storage->size_bytes(), original->size_bytes());

  auto cpu_storage_typed = dynamic_cast<CPUStorage<int> *>(cpu_storage.get());
  ASSERT_NE(cpu_storage_typed, nullptr);

  const auto &vec = cpu_storage_typed->vector();
  for (size_t i = 0; i < test_size; ++i) {
    EXPECT_EQ(vec[i], static_cast<int>(i + 1));
  }
}

// Cross-device transfer tests
class CrossDeviceTest : public StorageTest {};

TEST_F(CrossDeviceTest, CPUToCUDATransfer) {
  if (!cuda_available) {
    GTEST_SKIP() << "CUDA not available";
  }

  auto cpu_storage = std::make_unique<CPUStorage<float>>(test_size);
  auto &vec = cpu_storage->vector();
  std::iota(vec.begin(), vec.end(), 1.0f);

  auto cuda_storage = cpu_storage->to_device(Device::cuda(0));

  EXPECT_EQ(cuda_storage->device(), Device::cuda(0));
  EXPECT_EQ(cuda_storage->size_bytes(), cpu_storage->size_bytes());

  std::vector<float> result(test_size);
  cudaMemcpy(result.data(), cuda_storage->data(), cuda_storage->size_bytes(),
             cudaMemcpyDeviceToHost);

  for (size_t i = 0; i < test_size; ++i) {
    EXPECT_EQ(result[i], static_cast<float>(i + 1));
  }
}

// Factory function tests
class FactoryTest : public StorageTest {};

TEST_F(FactoryTest, MakeStorageCPU) {
  auto storage = make_storage<double>(test_size, Device::cpu());

  EXPECT_EQ(storage->device(), Device::cpu());
  EXPECT_EQ(storage->size_bytes(), test_size * sizeof(double));

  auto cpu_storage = dynamic_cast<CPUStorage<double> *>(storage.get());
  EXPECT_NE(cpu_storage, nullptr);
}

TEST_F(FactoryTest, MakeStorageCUDA) {
  if (!cuda_available) {
    GTEST_SKIP() << "CUDA not available";
  }

  auto storage = make_storage<float>(test_size, Device::cuda(0));

  EXPECT_EQ(storage->device(), Device::cuda(0));
  EXPECT_EQ(storage->size_bytes(), test_size * sizeof(float));

  auto cuda_storage = dynamic_cast<CUDAStorage<float> *>(storage.get());
  EXPECT_NE(cuda_storage, nullptr);
}

// Error handling tests
class ErrorHandlingTest : public StorageTest {};

TEST_F(ErrorHandlingTest, CUDAMoveAssignmentDifferentDevices) {
  if (!cuda_available) {
    GTEST_SKIP() << "CUDA not available";
  }

  int device_count = 0;
  cudaGetDeviceCount(&device_count);

  if (device_count < 2) {
    GTEST_SKIP() << "Need at least 2 CUDA devices for this test";
  }

  auto storage1 = std::make_unique<CUDAStorage<float>>(test_size, 0);
  auto storage2 = std::make_unique<CUDAStorage<float>>(test_size, 1);

  EXPECT_THROW(*storage1 = std::move(*storage2), std::runtime_error);
}

// Performance/Stress tests
class StressTest : public StorageTest {};

TEST_F(StressTest, LargeAllocation) {
  constexpr size_t large_size = 1000000; // 1M elements

  auto storage = std::make_unique<CPUStorage<float>>(large_size);
  EXPECT_EQ(storage->size_bytes(), large_size * sizeof(float));

  auto &vec = storage->vector();
  std::fill(vec.begin(), vec.end(), 42.0f);

  EXPECT_EQ(vec[0], 42.0f);
  EXPECT_EQ(vec[large_size - 1], 42.0f);
}

TEST_F(StressTest, MultipleClones) {
  auto original = std::make_unique<CPUStorage<int>>(test_size);
  auto &vec = original->vector();
  std::iota(vec.begin(), vec.end(), 1);

  std::vector<std::unique_ptr<Storage>> clones;
  for (int i = 0; i < 10; ++i) {
    clones.push_back(original->clone());
  }

  for (const auto &clone : clones) {
    EXPECT_EQ(clone->size_bytes(), original->size_bytes());
    EXPECT_EQ(clone->device(), original->device());

    auto clone_cpu = dynamic_cast<CPUStorage<int> *>(clone.get());
    ASSERT_NE(clone_cpu, nullptr);

    const auto &clone_vec = clone_cpu->vector();
    for (size_t j = 0; j < test_size; ++j) {
      EXPECT_EQ(clone_vec[j], static_cast<int>(j + 1));
    }
  }
}
