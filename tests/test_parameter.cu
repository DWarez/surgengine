#include <cmath>
#include <core/nn/parameter.cuh>
#include <core/tensor.cuh>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <sstream>

using namespace surgengine;
using namespace surgengine::nn;

class ParameterTest : public ::testing::Test {
protected:
  void SetUp() override {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    cuda_available_ = (device_count > 0);
    test_shape = {2, 3};
    test_name = "test_param";
  }

  std::vector<int> test_shape;
  std::string test_name;
  bool cuda_available_ = false;
};

// Basic Construction Tests
TEST_F(ParameterTest, DefaultConstruction) {
  Parameter<float> param;

  EXPECT_EQ(param.name(), "");
  EXPECT_TRUE(param.device().is_cpu());
  EXPECT_EQ(param.data().shape(), (std::vector<int64_t>{1, 1}));
  EXPECT_EQ(param.data().numel(), 1);
  EXPECT_TRUE(param.data().device().is_cpu());
}

TEST_F(ParameterTest, ConstructionWithName) {
  Parameter<float> param(test_name);

  EXPECT_EQ(param.name(), test_name);
  EXPECT_TRUE(param.device().is_cpu());
  EXPECT_EQ(param.data().shape(), (std::vector<int64_t>{1, 1}));
  EXPECT_EQ(param.data().numel(), 1);
}

TEST_F(ParameterTest, ConstructionWithDevice) {
  Parameter<float> param("", Device::cpu());

  EXPECT_EQ(param.name(), "");
  EXPECT_TRUE(param.device().is_cpu());
  EXPECT_TRUE(param.data().device().is_cpu());
}

TEST_F(ParameterTest, ConstructionWithNameAndDevice) {
  Parameter<float> param(test_name, Device::cpu());

  EXPECT_EQ(param.name(), test_name);
  EXPECT_TRUE(param.device().is_cpu());
  EXPECT_TRUE(param.data().device().is_cpu());
}

TEST_F(ParameterTest, ConstructionWithTensor) {
  FloatTensor tensor(test_shape);
  tensor.fill_(5.0f);

  Parameter<float> param(tensor, test_name);

  EXPECT_EQ(param.name(), test_name);
  EXPECT_TRUE(param.device().is_cpu());
  EXPECT_EQ(param.data().shape(), (std::vector<int64_t>{2, 3}));
  EXPECT_EQ(param.data().numel(), 6);

  // Verify data is copied/shared correctly
  EXPECT_FLOAT_EQ(param.data().data()[0], 5.0f);
}

TEST_F(ParameterTest, ConstructionWithTensorAndDevice) {
  FloatTensor tensor(test_shape);
  tensor.fill_(7.0f);

  Parameter<float> param(tensor, test_name, Device::cpu());

  EXPECT_EQ(param.name(), test_name);
  EXPECT_TRUE(param.device().is_cpu());
  EXPECT_EQ(param.data().shape(), (std::vector<int64_t>{2, 3}));
  EXPECT_FLOAT_EQ(param.data().data()[0], 7.0f);
}

// Data Access Tests
TEST_F(ParameterTest, DataAccess) {
  FloatTensor tensor(test_shape);
  tensor.fill_(3.14f);

  Parameter<float> param(tensor, test_name);

  // Test non-const data access
  Tensor<float> &data_ref = param.data();
  EXPECT_EQ(data_ref.numel(), 6);
  EXPECT_FLOAT_EQ(data_ref.data()[0], 3.14f);

  // Test const data access
  const Parameter<float> &const_param = param;
  const Tensor<float> &const_data_ref = const_param.data();
  EXPECT_EQ(const_data_ref.numel(), 6);
  EXPECT_FLOAT_EQ(const_data_ref.data()[0], 3.14f);
}

TEST_F(ParameterTest, DataModification) {
  Parameter<float> param(test_name);

  // Modify data through the parameter
  param.data().fill_(2.71f);
  EXPECT_FLOAT_EQ(param.data().data()[0], 2.71f);

  // Modify the tensor directly
  param.data().data()[0] = 1.41f;
  EXPECT_FLOAT_EQ(param.data().data()[0], 1.41f);
}

// Name Management Tests
TEST_F(ParameterTest, NameManagement) {
  Parameter<float> param;

  EXPECT_EQ(param.name(), "");

  param.set_name(test_name);
  EXPECT_EQ(param.name(), test_name);

  param.set_name("new_name");
  EXPECT_EQ(param.name(), "new_name");

  param.set_name("");
  EXPECT_EQ(param.name(), "");
}

// Device Management Tests
TEST_F(ParameterTest, DeviceConsistency) {
  Parameter<float> cpu_param(test_name, Device::cpu());

  EXPECT_TRUE(cpu_param.device().is_cpu());
  EXPECT_TRUE(cpu_param.data().device().is_cpu());
}

TEST_F(ParameterTest, CudaDeviceConsistency) {
  if (!cuda_available_) {
    GTEST_SKIP() << "CUDA not available";
  }

  Parameter<float> cuda_param(test_name, Device::cuda());

  EXPECT_TRUE(cuda_param.device().is_cuda());
  EXPECT_TRUE(cuda_param.data().device().is_cuda());
}

// Initialization Method Tests
TEST_F(ParameterTest, ZeroInitialization) {
  FloatTensor tensor(test_shape);
  tensor.fill_(99.0f); // Fill with non-zero values first

  Parameter<float> param(tensor, test_name);

  // Test method chaining
  Parameter<float> &result = param.zero_();
  EXPECT_EQ(&result, &param); // Should return reference to self

  // Verify all values are zero
  for (size_t i = 0; i < param.data().numel(); ++i) {
    EXPECT_FLOAT_EQ(param.data().data()[i], 0.0f);
  }
}

TEST_F(ParameterTest, FillInitialization) {
  FloatTensor tensor(test_shape);
  Parameter<float> param(tensor, test_name);

  const float fill_value = 42.0f;
  Parameter<float> &result = param.fill_(fill_value);
  EXPECT_EQ(&result, &param);

  // Verify all values are set to fill_value
  for (size_t i = 0; i < param.data().numel(); ++i) {
    EXPECT_FLOAT_EQ(param.data().data()[i], fill_value);
  }
}

TEST_F(ParameterTest, OnesInitialization) {
  FloatTensor tensor(test_shape);
  Parameter<float> param(tensor, test_name);

  Parameter<float> &result = param.ones_();
  EXPECT_EQ(&result, &param);

  // Verify all values are one
  for (size_t i = 0; i < param.data().numel(); ++i) {
    EXPECT_FLOAT_EQ(param.data().data()[i], 1.0f);
  }
}

TEST_F(ParameterTest, UniformInitialization) {
  FloatTensor tensor({100, 100}); // Larger tensor for better statistics
  Parameter<float> param(tensor, test_name);

  const float low = -2.0f;
  const float high = 3.0f;

  Parameter<float> &result = param.uniform_(low, high);
  EXPECT_EQ(&result, &param);

  // Verify values are in range (basic sanity check)
  for (size_t i = 0; i < param.data().numel(); ++i) {
    float value = param.data().data()[i];
    EXPECT_GE(value, low);
    EXPECT_LT(value, high);
  }
}

TEST_F(ParameterTest, UniformInitializationDefaults) {
  FloatTensor tensor({10, 10});
  Parameter<float> param(tensor, test_name);

  param.uniform_(); // Use default parameters

  // Verify values are in default range [-1, 1)
  for (size_t i = 0; i < param.data().numel(); ++i) {
    float value = param.data().data()[i];
    EXPECT_GE(value, -1.0f);
    EXPECT_LT(value, 1.0f);
  }
}

TEST_F(ParameterTest, NormalInitialization) {
  FloatTensor tensor({100, 100}); // Larger tensor for better statistics
  Parameter<float> param(tensor, test_name);

  const float mean = 1.0f;
  const float std = 0.5f;

  Parameter<float> &result = param.normal_(mean, std);
  EXPECT_EQ(&result, &param);

  // Basic sanity check - values should be roughly centered around mean
  float sum = 0.0f;
  for (size_t i = 0; i < param.data().numel(); ++i) {
    sum += param.data().data()[i];
  }
  float actual_mean = sum / param.data().numel();

  // Should be close to expected mean (within reasonable tolerance)
  EXPECT_NEAR(actual_mean, mean, 0.1f);
}

TEST_F(ParameterTest, NormalInitializationDefaults) {
  FloatTensor tensor({100, 100});
  Parameter<float> param(tensor, test_name);

  param.normal_(); // Use default parameters (mean=0, std=1)

  // Basic sanity check for default normal distribution
  float sum = 0.0f;
  for (size_t i = 0; i < param.data().numel(); ++i) {
    sum += param.data().data()[i];
  }
  float actual_mean = sum / param.data().numel();

  // Should be close to 0 (within reasonable tolerance)
  EXPECT_NEAR(actual_mean, 0.0f, 0.1f);
}

TEST_F(ParameterTest, XavierUniformInitialization) {
  FloatTensor tensor({100, 100});
  Parameter<float> param(tensor, test_name);

  Parameter<float> &result = param.xavier_uniform_();
  EXPECT_EQ(&result, &param);

  // Xavier uniform should initialize values in a specific range
  // The exact range depends on the implementation, but values should be
  // reasonable
  bool has_positive = false;
  bool has_negative = false;

  for (size_t i = 0; i < param.data().numel(); ++i) {
    float value = param.data().data()[i];
    if (value > 0)
      has_positive = true;
    if (value < 0)
      has_negative = true;

    // Should be in a reasonable range (not too large)
    EXPECT_LT(std::abs(value), 10.0f);
  }

  // Should have both positive and negative values
  EXPECT_TRUE(has_positive);
  EXPECT_TRUE(has_negative);
}

// Method Chaining Tests
TEST_F(ParameterTest, MethodChaining) {
  FloatTensor tensor(test_shape);
  Parameter<float> param(tensor, test_name);

  // Test chaining multiple initialization methods
  Parameter<float> &result = param.zero_().fill_(5.0f).ones_().zero_();

  EXPECT_EQ(&result, &param);

  // Final state should be all zeros
  for (size_t i = 0; i < param.data().numel(); ++i) {
    EXPECT_FLOAT_EQ(param.data().data()[i], 0.0f);
  }
}

// CUDA-specific Tests
TEST_F(ParameterTest, CudaInitialization) {
  if (!cuda_available_) {
    GTEST_SKIP() << "CUDA not available";
  }

  FloatTensor cuda_tensor(test_shape, Device::cuda());
  Parameter<float> param(cuda_tensor, test_name, Device::cuda());

  param.fill_(7.0f);

  // Copy to CPU to verify
  FloatTensor cpu_tensor = param.data().cpu();
  for (size_t i = 0; i < cpu_tensor.numel(); ++i) {
    EXPECT_FLOAT_EQ(cpu_tensor.data()[i], 7.0f);
  }
}

TEST_F(ParameterTest, CudaZeroInitialization) {
  if (!cuda_available_) {
    GTEST_SKIP() << "CUDA not available";
  }

  FloatTensor cuda_tensor(test_shape, Device::cuda());
  Parameter<float> param(cuda_tensor, test_name, Device::cuda());

  param.zero_();

  // Copy to CPU to verify
  FloatTensor cpu_tensor = param.data().cpu();
  for (size_t i = 0; i < cpu_tensor.numel(); ++i) {
    EXPECT_FLOAT_EQ(cpu_tensor.data()[i], 0.0f);
  }
}

// Template Tests
template <typename T> class TypedParameterTest : public ParameterTest {};

using TestedTypes = ::testing::Types<float, double, int32_t>;
TYPED_TEST_SUITE(TypedParameterTest, TestedTypes);

TYPED_TEST(TypedParameterTest, TypeSpecificOperations) {
  using ParamType = Parameter<TypeParam>;

  std::vector<int> shape = {2, 2};
  Tensor<TypeParam> tensor(shape);
  ParamType param(tensor, "typed_param");

  EXPECT_EQ(param.data().numel(), 4);

  // Test zero initialization
  param.zero_();
  EXPECT_EQ(param.data().data()[0], TypeParam{0});

  // Test fill initialization
  param.fill_(TypeParam{42});
  EXPECT_EQ(param.data().data()[0], TypeParam{42});
  EXPECT_EQ(param.data().data()[3], TypeParam{42});

  // Test ones initialization
  param.ones_();
  EXPECT_EQ(param.data().data()[0], TypeParam{1});
  EXPECT_EQ(param.data().data()[3], TypeParam{1});
}

// Integration Tests
TEST_F(ParameterTest, CompleteWorkflow) {
  // Create a parameter with a specific tensor
  FloatTensor tensor({3, 4});
  Parameter<float> param(tensor, "weight", Device::cpu());

  // Initialize with Xavier uniform
  param.xavier_uniform_();

  // Verify basic properties
  EXPECT_EQ(param.name(), "weight");
  EXPECT_TRUE(param.device().is_cpu());
  EXPECT_EQ(param.data().numel(), 12);

  // Change name
  param.set_name("bias");
  EXPECT_EQ(param.name(), "bias");

  // Reinitialize with specific value
  param.fill_(0.1f);
  for (size_t i = 0; i < param.data().numel(); ++i) {
    EXPECT_FLOAT_EQ(param.data().data()[i], 0.1f);
  }

  // Test method chaining in workflow
  param.zero_().ones_().fill_(2.0f);
  for (size_t i = 0; i < param.data().numel(); ++i) {
    EXPECT_FLOAT_EQ(param.data().data()[i], 2.0f);
  }
}

TEST_F(ParameterTest, ParameterWithDifferentShapes) {
  // Test with 1D tensor
  Parameter<float> param1d(FloatTensor({10}), "1d_param");
  EXPECT_EQ(param1d.data().numel(), 10);
  param1d.fill_(1.0f);

  // Test with 3D tensor
  Parameter<float> param3d(FloatTensor({2, 3, 4}), "3d_param");
  EXPECT_EQ(param3d.data().numel(), 24);
  param3d.fill_(2.0f);

  // Test with scalar (single element)
  Parameter<float> param_scalar(FloatTensor({1}), "scalar_param");
  EXPECT_EQ(param_scalar.data().numel(), 1);
  param_scalar.fill_(3.0f);
  EXPECT_FLOAT_EQ(param_scalar.data().data()[0], 3.0f);
}