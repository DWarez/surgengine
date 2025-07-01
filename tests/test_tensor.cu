#include "core/tensor.cuh"
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <sstream>

using namespace surgengine;

class TensorTest : public ::testing::Test {
protected:
  void SetUp() override {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    cuda_available_ = (device_count > 0);
  }

  bool cuda_available_ = false;
};

TEST_F(TensorTest, BasicConstruction) {
  FloatTensor tensor({2, 3, 4});

  EXPECT_EQ(tensor.shape(), (std::vector<int>{2, 3, 4}));
  EXPECT_EQ(tensor.numel(), 24);
  EXPECT_TRUE(tensor.device().is_cpu());
  EXPECT_TRUE(tensor.is_contiguous());
  EXPECT_NE(tensor.data(), nullptr);
}

TEST_F(TensorTest, EmptyTensor) {
  FloatTensor empty_tensor({});

  EXPECT_EQ(empty_tensor.numel(), 1); // Scalar
  EXPECT_TRUE(empty_tensor.is_contiguous());

  FloatTensor zero_size({0});
  EXPECT_EQ(zero_size.numel(), 0);
}

TEST_F(TensorTest, StrideComputation) {
  FloatTensor tensor({2, 3, 4});
  auto strides = tensor.strides();

  EXPECT_EQ(strides, (std::vector<int>{12, 4, 1})); // [3*4, 4, 1]
}

TEST_F(TensorTest, ZerosFactory) {
  auto tensor = FloatTensor::zeros({3, 3});

  EXPECT_EQ(tensor.numel(), 9);
  for (size_t i = 0; i < tensor.numel(); ++i) {
    EXPECT_FLOAT_EQ(tensor.data()[i], 0.0f);
  }
}

TEST_F(TensorTest, EmptyFactory) {
  auto tensor = FloatTensor::empty({2, 2});

  EXPECT_EQ(tensor.numel(), 4);
  EXPECT_NE(tensor.data(), nullptr);
}

TEST_F(TensorTest, FillOperation) {
  FloatTensor tensor({2, 3});
  tensor.fill_(3.14f);

  for (size_t i = 0; i < tensor.numel(); ++i) {
    EXPECT_FLOAT_EQ(tensor.data()[i], 3.14f);
  }
}

TEST_F(TensorTest, ZeroOperation) {
  FloatTensor tensor({2, 2});
  tensor.fill_(99.0f);
  tensor.zero_();

  for (size_t i = 0; i < tensor.numel(); ++i) {
    EXPECT_FLOAT_EQ(tensor.data()[i], 0.0f);
  }
}

TEST_F(TensorTest, ConstCorrectness) {
  const FloatTensor tensor({2, 2});

  const float *const_ptr = tensor.data();
  EXPECT_NE(const_ptr, nullptr);

  EXPECT_EQ(tensor.numel(), 4);
  EXPECT_TRUE(tensor.is_contiguous());
}

TEST_F(TensorTest, ViewReshape) {
  FloatTensor original({2, 6});
  original.fill_(42.0f);

  auto reshaped = original.view({3, 4});

  EXPECT_EQ(reshaped.shape(), (std::vector<int>{3, 4}));
  EXPECT_EQ(reshaped.numel(), 12);
  EXPECT_TRUE(original.shares_storage(reshaped));

  reshaped.fill_(7.0f);
  EXPECT_FLOAT_EQ(original.data()[0], 7.0f);
}

TEST_F(TensorTest, ViewInvalidReshape) {
  FloatTensor tensor({2, 3});

  EXPECT_THROW(tensor.view({2, 4}), std::invalid_argument); // Wrong size
}

TEST_F(TensorTest, SliceOperation) {
  FloatTensor tensor({4, 3});

  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 3; ++c) {
      tensor.data()[r * 3 + c] = static_cast<float>(r);
    }
  }

  auto slice = tensor.slice(0, 1, 3); // Rows 1-2

  EXPECT_EQ(slice.shape(), (std::vector<int>{2, 3}));
  EXPECT_TRUE(tensor.shares_storage(slice));
  EXPECT_FLOAT_EQ(slice.data()[0], 1.0f); // First element should be from row 1
  EXPECT_FLOAT_EQ(slice.data()[3], 2.0f); // First element of row 2
}

TEST_F(TensorTest, SliceInvalidDimension) {
  FloatTensor tensor({2, 3});

  EXPECT_THROW(tensor.slice(2, 0, 1), std::out_of_range); // Dim 2 doesn't exist
}

TEST_F(TensorTest, StorageSharing) {
  FloatTensor original({3, 3});
  original.fill_(1.0f);

  auto view1 = original.view({9});
  auto view2 = original.slice(0, 0, 2);

  EXPECT_TRUE(original.shares_storage(view1));
  EXPECT_TRUE(original.shares_storage(view2));
  EXPECT_TRUE(view1.shares_storage(view2));

  view1.fill_(2.0f);
  EXPECT_FLOAT_EQ(original.data()[0], 2.0f);
  EXPECT_FLOAT_EQ(view2.data()[0], 2.0f);
}

TEST_F(TensorTest, MemoryUsage) {
  FloatTensor tensor({10, 10});

  EXPECT_EQ(tensor.memory_usage(), 100 * sizeof(float));

  auto view = tensor.view({20, 5});
  EXPECT_EQ(view.memory_usage(), 100 * sizeof(float)); // Same storage
}

TEST_F(TensorTest, CudaBasicOperations) {
  if (!cuda_available_) {
    GTEST_SKIP() << "CUDA not available";
  }

  FloatTensor gpu_tensor({5, 5}, Device::cuda());

  EXPECT_TRUE(gpu_tensor.device().is_cuda());
  EXPECT_EQ(gpu_tensor.numel(), 25);
  EXPECT_NE(gpu_tensor.data(), nullptr);

  gpu_tensor.fill_(3.14f);

  FloatTensor cpu_result({5, 5});
  cudaMemcpy(cpu_result.data(), gpu_tensor.data(), 25 * sizeof(float),
             cudaMemcpyDeviceToHost);

  for (size_t i = 0; i < 25; ++i) {
    EXPECT_FLOAT_EQ(cpu_result.data()[i], 3.14f);
  }
}

TEST_F(TensorTest, CudaZeros) {
  if (!cuda_available_) {
    GTEST_SKIP() << "CUDA not available";
  }

  auto gpu_tensor = FloatTensor::zeros({3, 3}, Device::cuda());

  EXPECT_TRUE(gpu_tensor.device().is_cuda());

  FloatTensor cpu_check({3, 3});
  cudaMemcpy(cpu_check.data(), gpu_tensor.data(), 9 * sizeof(float),
             cudaMemcpyDeviceToHost);

  for (size_t i = 0; i < 9; ++i) {
    EXPECT_FLOAT_EQ(cpu_check.data()[i], 0.0f);
  }
}

TEST_F(TensorTest, PrintBasic) {
  FloatTensor tensor({2, 3});
  tensor.fill_(2.5f);

  std::ostringstream oss;
  oss << tensor;
  std::string output = oss.str();

  EXPECT_TRUE(output.find("shape=[2, 3]") != std::string::npos);
  EXPECT_TRUE(output.find("device=cpu") != std::string::npos);
  EXPECT_TRUE(output.find("dtype=float32") != std::string::npos);
  EXPECT_TRUE(output.find("numel=6") != std::string::npos);

  EXPECT_TRUE(output.find("2.5000") != std::string::npos);
}

TEST_F(TensorTest, Print1D) {
  FloatTensor tensor({5});
  float *data = tensor.data();
  for (int i = 0; i < 5; ++i) {
    data[i] = static_cast<float>(i + 1);
  }

  std::ostringstream oss;
  oss << tensor;
  std::string output = oss.str();

  EXPECT_TRUE(output.find("[1.0000, 2.0000, 3.0000, 4.0000, 5.0000]") !=
              std::string::npos);
}

TEST_F(TensorTest, PrintEmpty) {
  FloatTensor empty_tensor({0});

  std::ostringstream oss;
  oss << empty_tensor;
  std::string output = oss.str();

  EXPECT_TRUE(output.find("numel=0") != std::string::npos);
  EXPECT_TRUE(output.find("[]") != std::string::npos);
}

TEST_F(TensorTest, PrintLargeTensorTruncation) {
  FloatTensor large_tensor({100});
  large_tensor.fill_(1.0f);

  std::ostringstream oss;
  oss << large_tensor;
  std::string output = oss.str();

  EXPECT_TRUE(output.find("...") != std::string::npos);
  EXPECT_TRUE(output.find("numel=100") != std::string::npos);
}

template <typename T> class TypedTensorTest : public TensorTest {};

using TestedTypes = ::testing::Types<float, double, int32_t>;
TYPED_TEST_SUITE(TypedTensorTest, TestedTypes);

TYPED_TEST(TypedTensorTest, TypeSpecificOperations) {
  using TensorType = Tensor<TypeParam>;

  TensorType tensor({3, 3});
  EXPECT_EQ(tensor.numel(), 9);

  tensor.zero_();
  EXPECT_EQ(tensor.data()[0], TypeParam{0});

  tensor.fill_(TypeParam{42});
  EXPECT_EQ(tensor.data()[0], TypeParam{42});
  EXPECT_EQ(tensor.data()[8], TypeParam{42});
}

TEST_F(TensorTest, ErrorHandling) {
  FloatTensor tensor({2, 3});

  EXPECT_THROW(tensor.slice(0, 5, 10), std::out_of_range);

  EXPECT_THROW(tensor.slice(3, 0, 1), std::out_of_range);

  EXPECT_THROW(tensor.view({7}), std::invalid_argument);
}

TEST_F(TensorTest, CompleteWorkflow) {
  auto original = FloatTensor::zeros({4, 6});
  original.fill_(1.0f);

  auto reshaped = original.view({6, 4});
  EXPECT_TRUE(original.shares_storage(reshaped));

  auto slice = reshaped.slice(0, 1, 5); // 4 rows
  EXPECT_EQ(slice.shape(), (std::vector<int>{4, 4}));

  slice.fill_(2.0f);

  EXPECT_FLOAT_EQ(original.data()[0], 1.0f); // First row unchanged
  EXPECT_FLOAT_EQ(original.data()[6], 2.0f); // Modified through slice
}