#include "core/storage.cuh"
#include <core/tensor.cuh>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <memory>

using namespace surgengine;

class StorageTest : public ::testing::Test {
protected:
  bool cuda_available_ = false;

  void SetUp() override {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    cuda_available_ = (device_count > 0);

    if (cuda_available_) {
      cudaSetDevice(0);
    }
  }

  void TearDown() override {
    if (cuda_available_)
      cudaDeviceReset();
  }
};

TEST_F(StorageTest, CPUStorageCreation) {
  auto storage = std::make_unique<CPUStorage<float>>(100);

  EXPECT_NE(storage->data(), nullptr);
  EXPECT_EQ(storage->size_bytes(), 100 * sizeof(float));
  EXPECT_TRUE(storage->device().is_cpu());
  EXPECT_EQ(storage->device().rank, 0);
}