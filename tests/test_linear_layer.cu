#include <core/device.cuh>
#include <core/nn/linear_layer.cuh>
#include <core/nn/module.cuh>
#include <core/nn/parameter.cuh>
#include <core/tensor.cuh>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <memory>
#include <random>
#include <vector>

using namespace surgengine::nn;
using namespace surgengine;

class LinearLayerTest : public ::testing::Test {
protected:
  void SetUp() override {
    std::srand(42);

    cpu_device_ = Device::cpu();

    // Check CUDA availability properly
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    cuda_available_ = (device_count > 0);

    if (cuda_available_) {
      cuda_device_ = Device::cuda(0);
    }

    linear_layer_ = std::make_unique<LinearLayer<float>>(
        5, 3, true, "test_linear", cpu_device_);
  }

  void TearDown() override { LinearLayer<float>::cleanup_cublas(); }

  FloatTensor createRandomTensor(const std::vector<int> &shape,
                                 const Device &device) {
    FloatTensor tensor(shape, Device::cpu());
    float *data = tensor.data();
    std::random_device rd;
    std::mt19937 gen(42);
    std::normal_distribution<float> dis(0.0f, 1.0f);

    size_t total_size = 1;
    for (int dim : shape) {
      total_size *= static_cast<size_t>(dim);
    }

    for (size_t i = 0; i < total_size; ++i) {
      data[i] = dis(gen);
    }

    // Return tensor on the correct device
    return device.is_cpu() ? tensor : tensor.to(device);
  }

  bool tensorsApproxEqual(const FloatTensor &a, const FloatTensor &b,
                          float tolerance = 1e-5f) {
    if (a.shape() != b.shape())
      return false;

    // Copy both tensors to CPU for comparison
    FloatTensor a_cpu = a.device().is_cpu() ? a : a.cpu();
    FloatTensor b_cpu = b.device().is_cpu() ? b : b.cpu();

    const float *a_data = a_cpu.data();
    const float *b_data = b_cpu.data();

    size_t total_size = 1;
    for (int64_t dim : a.shape()) {
      total_size *= static_cast<size_t>(dim);
    }

    for (size_t i = 0; i < total_size; ++i) {
      if (std::abs(a_data[i] - b_data[i]) > tolerance) {
        return false;
      }
    }
    return true;
  }

  Device cpu_device_;
  Device cuda_device_;
  bool cuda_available_ = false;
  std::unique_ptr<LinearLayer<float>> linear_layer_;
};

TEST_F(LinearLayerTest, Constructor) {
  LinearLayer<float> default_layer(10, 5);
  EXPECT_EQ(default_layer.device().type, Device::cpu().type);
  EXPECT_EQ(default_layer.in_features(), 10);
  EXPECT_EQ(default_layer.out_features(), 5);
  EXPECT_TRUE(default_layer.has_bias());

  if (cuda_available_) {
    LinearLayer<float> custom_layer(8, 4, false, "custom_linear", cuda_device_);
    EXPECT_EQ(custom_layer.device().type, cuda_device_.type);
    EXPECT_EQ(custom_layer.name(), "custom_linear");
    EXPECT_EQ(custom_layer.in_features(), 8);
    EXPECT_EQ(custom_layer.out_features(), 4);
    EXPECT_FALSE(custom_layer.has_bias());
  }
}

TEST_F(LinearLayerTest, ConstructorWithBias) {
  LinearLayer<float> layer(5, 3, true, "bias_layer", cpu_device_);
  EXPECT_TRUE(layer.has_bias());

  auto params = layer.parameters();
  EXPECT_EQ(params.size(), 2);
  auto weight_param = layer.get_parameter("weight");
  auto bias_param = layer.get_parameter("bias");
  EXPECT_NE(weight_param, nullptr);
  EXPECT_NE(bias_param, nullptr);
}

TEST_F(LinearLayerTest, ConstructorWithoutBias) {
  LinearLayer<float> layer(5, 3, false, "no_bias_layer", cpu_device_);
  EXPECT_FALSE(layer.has_bias());

  auto params = layer.parameters();
  EXPECT_EQ(params.size(), 1);

  // Check that we have weight but no bias using get_parameter
  auto weight_param = layer.get_parameter("weight");
  auto bias_param = layer.get_parameter("bias");
  EXPECT_NE(weight_param, nullptr);
  EXPECT_EQ(bias_param, nullptr);
}

TEST_F(LinearLayerTest, Forward1DInput) {
  FloatTensor input({5}, cpu_device_);
  float *input_data = input.data();
  for (int i = 0; i < 5; ++i) {
    input_data[i] = static_cast<float>(i + 1);
  }

  FloatTensor output = linear_layer_->forward(input);

  EXPECT_EQ(output.shape().size(), 1);
  EXPECT_EQ(output.shape()[0], 3);
}

TEST_F(LinearLayerTest, Forward2DInput) {
  FloatTensor input({2, 5}, cpu_device_);
  float *input_data = input.data();
  for (int i = 0; i < 10; ++i) {
    input_data[i] = static_cast<float>(i + 1);
  }

  FloatTensor output = linear_layer_->forward(input);

  EXPECT_EQ(output.shape().size(), 2);
  EXPECT_EQ(output.shape()[0], 2);
  EXPECT_EQ(output.shape()[1], 3);
}

TEST_F(LinearLayerTest, WrongInputFeatureSize) {
  FloatTensor wrong_input_1d({4}, cpu_device_);
  EXPECT_THROW(linear_layer_->forward(wrong_input_1d), std::runtime_error);

  FloatTensor wrong_input_2d({2, 4}, cpu_device_);
  EXPECT_THROW(linear_layer_->forward(wrong_input_2d), std::runtime_error);
}

TEST_F(LinearLayerTest, LinearTransformationCorrectness) {
  LinearLayer<float> layer(2, 1, false, "math_test", cpu_device_);

  auto weight_param = layer.get_parameter("weight");
  ASSERT_NE(weight_param, nullptr);
  float *weight_data = weight_param->data().data();
  weight_data[0] = 2.0f;
  weight_data[1] = 3.0f;

  FloatTensor input({2}, cpu_device_);
  float *input_data = input.data();
  input_data[0] = 1.0f;
  input_data[1] = 2.0f;

  FloatTensor output = layer.forward(input);

  // Expected: 2.0 * 1.0 + 3.0 * 2.0 = 8.0
  float expected = 8.0f;
  EXPECT_NEAR(output.data()[0], expected, 1e-6f);
}

TEST_F(LinearLayerTest, BiasAddition) {
  LinearLayer<float> layer(2, 1, true, "bias_test", cpu_device_);

  auto weight_param = layer.get_parameter("weight");
  auto bias_param = layer.get_parameter("bias");

  ASSERT_NE(weight_param, nullptr);
  ASSERT_NE(bias_param, nullptr);

  float *weight_data = weight_param->data().data();
  weight_data[0] = 2.0f;
  weight_data[1] = 3.0f;

  float *bias_data = bias_param->data().data();
  bias_data[0] = 1.0f;

  FloatTensor input({2}, cpu_device_);
  float *input_data = input.data();
  input_data[0] = 1.0f;
  input_data[1] = 2.0f;

  FloatTensor output = layer.forward(input);

  // Expected: 2.0 * 1.0 + 3.0 * 2.0 + 1.0 = 9.0
  float expected = 9.0f;
  EXPECT_NEAR(output.data()[0], expected, 1e-6f);
}

TEST_F(LinearLayerTest, BatchProcessing) {
  FloatTensor batch_input({3, 5}, cpu_device_);
  float *input_data = batch_input.data();
  for (int i = 0; i < 15; ++i) {
    input_data[i] = static_cast<float>(i + 1);
  }

  FloatTensor output = linear_layer_->forward(batch_input);

  EXPECT_EQ(output.shape().size(), 2);
  EXPECT_EQ(output.shape()[0], 3); // batch size
  EXPECT_EQ(output.shape()[1], 3); // output features
}

// CUDA-specific tests
TEST_F(LinearLayerTest, CUDAForward) {
  if (!cuda_available_) {
    GTEST_SKIP() << "CUDA not available";
  }

  LinearLayer<float> cuda_layer(5, 3, true, "cuda_layer", cuda_device_);
  FloatTensor input = createRandomTensor({10, 5}, cuda_device_);
  FloatTensor output = cuda_layer.forward(input);

  EXPECT_EQ(output.shape()[0], 10);
  EXPECT_EQ(output.shape()[1], 3);
  EXPECT_TRUE(output.device().is_cuda());
}

TEST_F(LinearLayerTest, CUDABatchProcessing) {
  if (!cuda_available_) {
    GTEST_SKIP() << "CUDA not available";
  }

  LinearLayer<float> cuda_layer(5, 3, true, "cuda_batch_layer", cuda_device_);
  FloatTensor input = createRandomTensor({4, 5}, cuda_device_);
  FloatTensor output = cuda_layer.forward(input);

  EXPECT_EQ(output.shape()[0], 4);
  EXPECT_EQ(output.shape()[1], 3);
  EXPECT_TRUE(output.device().is_cuda());
}

TEST_F(LinearLayerTest, CUDALinearTransformation) {
  if (!cuda_available_) {
    GTEST_SKIP() << "CUDA not available";
  }

  LinearLayer<float> cuda_layer(2, 1, false, "cuda_math_test", cuda_device_);

  auto weight_param = cuda_layer.get_parameter("weight");
  ASSERT_NE(weight_param, nullptr);

  // Set weights on CPU then move to CUDA
  FloatTensor cpu_weight({1, 2}, Device::cpu());
  cpu_weight.data()[0] = 2.0f;
  cpu_weight.data()[1] = 3.0f;
  FloatTensor cuda_weight = cpu_weight.to(cuda_device_);

  // Copy data to parameter (this might need adjustment based on actual API)
  cudaMemcpy(weight_param->data().data(), cuda_weight.data(), 2 * sizeof(float),
             cudaMemcpyDeviceToDevice);

  FloatTensor input = createRandomTensor({2}, cuda_device_);
  input.data()[0] = 1.0f;
  input.data()[1] = 2.0f;

  FloatTensor output = cuda_layer.forward(input);

  // Copy result back to CPU for verification
  FloatTensor cpu_output = output.cpu();

  // Expected: 2.0 * 1.0 + 3.0 * 2.0 = 8.0
  float expected = 8.0f;
  EXPECT_NEAR(cpu_output.data()[0], expected, 1e-6f);
}

TEST_F(LinearLayerTest, LargeInputs) {
  LinearLayer<float> large_layer(1000, 500, true, "large_layer", cpu_device_);

  FloatTensor large_input = createRandomTensor({100, 1000}, cpu_device_);

  EXPECT_NO_THROW({
    FloatTensor output = large_layer.forward(large_input);
    EXPECT_EQ(output.shape()[0], 100);
    EXPECT_EQ(output.shape()[1], 500);
  });
}

TEST_F(LinearLayerTest, CUDALargeInputs) {
  if (!cuda_available_) {
    GTEST_SKIP() << "CUDA not available";
  }

  LinearLayer<float> large_cuda_layer(1000, 500, true, "large_cuda_layer",
                                      cuda_device_);

  FloatTensor large_input = createRandomTensor({100, 1000}, cuda_device_);

  EXPECT_NO_THROW({
    FloatTensor output = large_cuda_layer.forward(large_input);
    EXPECT_EQ(output.shape()[0], 100);
    EXPECT_EQ(output.shape()[1], 500);
    EXPECT_TRUE(output.device().is_cuda());
  });
}

TEST_F(LinearLayerTest, ParameterInitialization) {
  LinearLayer<float> layer(10, 5, true, "init_test", cpu_device_);

  auto weight_param = layer.get_parameter("weight");
  auto bias_param = layer.get_parameter("bias");

  ASSERT_NE(weight_param, nullptr);
  ASSERT_NE(bias_param, nullptr);

  // Check weight shape
  EXPECT_EQ(weight_param->data().shape(), (std::vector<int64_t>{5, 10}));

  // Check bias shape
  EXPECT_EQ(bias_param->data().shape(), (std::vector<int64_t>{5}));

  // Verify parameters are on correct device
  EXPECT_TRUE(weight_param->data().device().is_cpu());
  EXPECT_TRUE(bias_param->data().device().is_cpu());
}

TEST_F(LinearLayerTest, CUDAParameterInitialization) {
  if (!cuda_available_) {
    GTEST_SKIP() << "CUDA not available";
  }

  LinearLayer<float> cuda_layer(10, 5, true, "cuda_init_test", cuda_device_);

  auto weight_param = cuda_layer.get_parameter("weight");
  auto bias_param = cuda_layer.get_parameter("bias");

  ASSERT_NE(weight_param, nullptr);
  ASSERT_NE(bias_param, nullptr);

  // Check weight shape
  EXPECT_EQ(weight_param->data().shape(), (std::vector<int64_t>{5, 10}));

  // Check bias shape
  EXPECT_EQ(bias_param->data().shape(), (std::vector<int64_t>{5}));

  // Verify parameters are on correct device
  EXPECT_TRUE(weight_param->data().device().is_cuda());
  EXPECT_TRUE(bias_param->data().device().is_cuda());
}

TEST_F(LinearLayerTest, ErrorHandling) {
  // Test with invalid dimensions
  EXPECT_THROW(LinearLayer<float>(0, 5), std::invalid_argument);
  EXPECT_THROW(LinearLayer<float>(5, 0), std::invalid_argument);

  // Test with negative dimensions
  EXPECT_THROW(LinearLayer<float>(-1, 5), std::invalid_argument);
  EXPECT_THROW(LinearLayer<float>(5, -1), std::invalid_argument);
}

TEST_F(LinearLayerTest, DeviceConsistency) {
  LinearLayer<float> cpu_layer(5, 3, true, "cpu_layer", cpu_device_);

  // Input on wrong device should throw or be handled appropriately
  if (cuda_available_) {
    FloatTensor cuda_input = createRandomTensor({5}, cuda_device_);
    // This should either throw or automatically move the tensor to the correct
    // device The exact behavior depends on the implementation
    EXPECT_NO_THROW({ FloatTensor output = cpu_layer.forward(cuda_input); });
  }
}