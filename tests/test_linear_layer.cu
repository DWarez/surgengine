#include <core/device.cuh>
#include <core/nn/linear_layer.cuh>
#include <core/nn/module.cuh>
#include <core/nn/parameter.cuh>
#include <core/tensor.cuh>
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
    if (Device::is_cuda_available()) {
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
      total_size *= dim;
    }

    for (size_t i = 0; i < total_size; ++i) {
      data[i] = dis(gen);
    }
    return device.is_cpu() ? tensor : tensor.to(Device::cuda());
  }

  bool tensorsApproxEqual(const FloatTensor &a, const FloatTensor &b,
                          float tolerance = 1e-5f) {
    if (a.shape() != b.shape())
      return false;

    const float *a_data = a.data();
    const float *b_data = b.data();

    size_t total_size = 1;
    for (int dim : a.shape()) {
      total_size *= dim;
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
  std::unique_ptr<LinearLayer<float>> linear_layer_;
};

TEST_F(LinearLayerTest, Constructor) {
  LinearLayer<float> default_layer(10, 5);
  EXPECT_EQ(default_layer.device().type, Device::cpu().type);
  EXPECT_EQ(default_layer.in_features(), 10);
  EXPECT_EQ(default_layer.out_features(), 5);
  EXPECT_TRUE(default_layer.has_bias());

  LinearLayer<float> custom_layer(8, 4, false, "custom_linear", cuda_device_);
  EXPECT_EQ(custom_layer.device().type, cuda_device_.type);
  EXPECT_EQ(custom_layer.name(), "custom_linear");
  EXPECT_EQ(custom_layer.in_features(), 8);
  EXPECT_EQ(custom_layer.out_features(), 4);
  EXPECT_FALSE(custom_layer.has_bias());
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
  float *weight_data = weight_param->data().data();
  weight_data[0] = 2.0f;
  weight_data[1] = 3.0f;

  FloatTensor input({2}, cpu_device_);
  float *input_data = input.data();
  input_data[0] = 1.0f;
  input_data[1] = 2.0f;

  FloatTensor output = layer.forward(input);

  float expected = 8.0f;
  EXPECT_NEAR(output.data()[0], expected, 1e-6f);
}

TEST_F(LinearLayerTest, BiasAddition) {
  LinearLayer<float> layer(2, 1, true, "bias_test", cpu_device_);

  auto weight_param = layer.get_parameter("weight");
  auto bias_param = layer.get_parameter("bias");

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
  if (!Device::is_cuda_available()) {
    GTEST_SKIP() << "CUDA not available";
  }
  LinearLayer<float> cuda_layer(5, 3, true, "cuda_layer", cuda_device_);
  Tensor<float> input = createRandomTensor({10, 5}, cuda_device_);
  Tensor<float> output = cuda_layer.forward(input);

  EXPECT_EQ(output.shape()[0], 10);
  EXPECT_EQ(output.shape()[1], 3);
}

TEST_F(LinearLayerTest, LargeInputs) {
  LinearLayer<float> large_layer(1000, 500, true, "large_layer", cpu_device_);

  Tensor<float> large_input = createRandomTensor({100, 1000}, cpu_device_);

  EXPECT_NO_THROW({
    Tensor<float> output = large_layer.forward(large_input);
    EXPECT_EQ(output.shape()[0], 100);
    EXPECT_EQ(output.shape()[1], 500);
  });
}
