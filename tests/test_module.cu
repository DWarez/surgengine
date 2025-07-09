#include "core/nn/parameter.cuh"
#include "core/tensor.cuh"
#include <core/device.cuh>
#include <core/nn/module.cuh>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <memory>
#include <sstream>

using namespace surgengine::nn;

// Concrete implementation of Module for testing
template <typename T> class TestModule : public Module<T> {
public:
  TestModule(const std::string &name = "", const Device &device = Device::cpu())
      : Module<T>(name, device) {}

  Tensor<T> forward(const Tensor<T> &input) override {
    return input; // Simple passthrough for testing
  }
};

class ModuleTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Check CUDA availability
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    cuda_available_ = (device_count > 0);

    module_ = std::make_unique<TestModule<float>>("test_module");
  }

  bool cuda_available_ = false;
  std::unique_ptr<TestModule<float>> module_;
};

TEST_F(ModuleTest, Constructor) {
  TestModule<float> default_module;
  EXPECT_TRUE(default_module.device().is_cpu());

  if (cuda_available_) {
    TestModule<float> named_module("custom", Device::cuda(0));
    EXPECT_TRUE(named_module.device().is_cuda());
    EXPECT_EQ(named_module.device().rank, 0);
    EXPECT_EQ(named_module.name(), "custom");
  }
}

TEST_F(ModuleTest, OperatorCall) {
  Tensor<float> input({2, 3});
  input.fill_(1.0f);

  Tensor<float> output = (*module_)(input);

  // Output should be same as input for our test implementation
  EXPECT_EQ(output.shape(), input.shape());
  EXPECT_EQ(output.numel(), input.numel());

  // Verify data is the same
  for (size_t i = 0; i < output.numel(); ++i) {
    EXPECT_FLOAT_EQ(output.data()[i], input.data()[i]);
  }
}

TEST_F(ModuleTest, DeviceGetter) {
  EXPECT_TRUE(module_->device().is_cpu());

  if (cuda_available_) {
    TestModule<float> cuda_module("cuda_test", Device::cuda(0));
    EXPECT_TRUE(cuda_module.device().is_cuda());
    EXPECT_EQ(cuda_module.device().rank, 0);
  }
}

TEST_F(ModuleTest, ToMethod) {
  if (!cuda_available_) {
    GTEST_SKIP() << "CUDA not available";
  }

  Device cuda_device = Device::cuda(0);
  Module<float> &result = module_->to(cuda_device);

  EXPECT_TRUE(module_->device().is_cuda());
  EXPECT_EQ(module_->device().rank, 0);
  EXPECT_EQ(&result, module_.get());
}

TEST_F(ModuleTest, ToMethodCPU) {
  // Test moving from CPU to CPU (should be no-op)
  Module<float> &result = module_->to(Device::cpu());

  EXPECT_TRUE(module_->device().is_cpu());
  EXPECT_EQ(&result, module_.get());
}

TEST_F(ModuleTest, CudaMethod) {
  if (!cuda_available_) {
    GTEST_SKIP() << "CUDA not available";
  }

  Module<float> &result = module_->cuda(1);

  EXPECT_TRUE(module_->device().is_cuda());
  EXPECT_EQ(module_->device().rank, 1);
  EXPECT_EQ(&result, module_.get());
}

TEST_F(ModuleTest, CudaMethodDefaultDevice) {
  if (!cuda_available_) {
    GTEST_SKIP() << "CUDA not available";
  }

  Module<float> &result = module_->cuda();

  EXPECT_TRUE(module_->device().is_cuda());
  EXPECT_EQ(module_->device().rank, 0);
  EXPECT_EQ(&result, module_.get());
}

TEST_F(ModuleTest, CpuMethod) {
  if (!cuda_available_) {
    GTEST_SKIP() << "CUDA not available";
  }

  module_->cuda(0);
  EXPECT_TRUE(module_->device().is_cuda());

  Module<float> &result = module_->cpu();

  EXPECT_TRUE(module_->device().is_cpu());
  EXPECT_EQ(&result, module_.get());
}

TEST_F(ModuleTest, CpuMethodFromCPU) {
  // Test moving from CPU to CPU (should be no-op)
  EXPECT_TRUE(module_->device().is_cpu());

  Module<float> &result = module_->cpu();

  EXPECT_TRUE(module_->device().is_cpu());
  EXPECT_EQ(&result, module_.get());
}

TEST_F(ModuleTest, NameGetterSetter) {
  EXPECT_EQ(module_->name(), "test_module");

  module_->set_name("new_name");
  EXPECT_EQ(module_->name(), "new_name");

  module_->set_name("");
  EXPECT_EQ(module_->name(), "");
}

TEST_F(ModuleTest, RegisterGetParameter) {
  auto param = std::make_shared<Parameter<float>>(Tensor<float>({2, 2}));
  module_->register_parameter("weight", param);

  auto retrieved = module_->get_parameter("weight");
  EXPECT_EQ(retrieved, param);

  auto null_param = module_->get_parameter("nonexistent");
  EXPECT_EQ(null_param, nullptr);
}

TEST_F(ModuleTest, RegisterGetParameterWithEmptyName) {
  auto param = std::make_shared<Parameter<float>>(Tensor<float>({2, 2}));

  // Test with empty parameter name
  EXPECT_NO_THROW(module_->register_parameter("", param));

  auto retrieved = module_->get_parameter("");
  EXPECT_EQ(retrieved, param);
}

TEST_F(ModuleTest, RegisterGetSubmodule) {
  auto submodule = std::make_shared<TestModule<float>>("sub");
  module_->register_module("sub", submodule);

  auto retrieved = module_->get_submodule("sub");
  EXPECT_EQ(retrieved, submodule);

  auto null_module = module_->get_submodule("nonexistent");
  EXPECT_EQ(null_module, nullptr);
}

TEST_F(ModuleTest, RegisterGetSubmoduleWithEmptyName) {
  auto submodule = std::make_shared<TestModule<float>>("sub");

  // Test with empty submodule name
  EXPECT_NO_THROW(module_->register_module("", submodule));

  auto retrieved = module_->get_submodule("");
  EXPECT_EQ(retrieved, submodule);
}

TEST_F(ModuleTest, ParametersMethod) {
  auto param1 = std::make_shared<Parameter<float>>(Tensor<float>({2, 2}));
  auto param2 = std::make_shared<Parameter<float>>(Tensor<float>({3, 3}));

  module_->register_parameter("weight1", param1);
  module_->register_parameter("weight2", param2);

  auto all_params = module_->parameters();
  EXPECT_EQ(all_params.size(), 2);

  // Verify the parameters are actually in the collection
  bool found_param1 = false, found_param2 = false;
  for (const auto &p : all_params) {
    if (p == param1)
      found_param1 = true;
    if (p == param2)
      found_param2 = true;
  }
  EXPECT_TRUE(found_param1);
  EXPECT_TRUE(found_param2);
}

TEST_F(ModuleTest, ParametersWithSubmodules) {
  auto param1 = std::make_shared<Parameter<float>>(Tensor<float>({2, 2}));
  auto submodule = std::make_shared<TestModule<float>>("sub");
  auto param2 = std::make_shared<Parameter<float>>(Tensor<float>({3, 3}));

  module_->register_parameter("weight1", param1);
  submodule->register_parameter("weight2", param2);
  module_->register_module("sub", submodule);

  auto all_params = module_->parameters();
  EXPECT_EQ(all_params.size(), 2);

  // Verify both parameters are included
  bool found_param1 = false, found_param2 = false;
  for (const auto &p : all_params) {
    if (p == param1)
      found_param1 = true;
    if (p == param2)
      found_param2 = true;
  }
  EXPECT_TRUE(found_param1);
  EXPECT_TRUE(found_param2);
}

TEST_F(ModuleTest, ParametersEmpty) {
  auto all_params = module_->parameters();
  EXPECT_EQ(all_params.size(), 0);
}

TEST_F(ModuleTest, NamedParametersMethod) {
  auto param1 = std::make_shared<Parameter<float>>(Tensor<float>({2, 2}));
  auto param2 = std::make_shared<Parameter<float>>(Tensor<float>({3, 3}));

  module_->register_parameter("weight1", param1);
  module_->register_parameter("weight2", param2);

  auto named_params = module_->named_parameters();
  EXPECT_EQ(named_params.size(), 2);
  EXPECT_NE(named_params.find("test_module.weight1"), named_params.end());
  EXPECT_NE(named_params.find("test_module.weight2"), named_params.end());

  // Verify the parameters are correct
  EXPECT_EQ(named_params["test_module.weight1"], param1);
  EXPECT_EQ(named_params["test_module.weight2"], param2);
}

TEST_F(ModuleTest, NamedParametersWithSubmodules) {
  auto param1 = std::make_shared<Parameter<float>>(Tensor<float>({2, 2}));
  auto submodule = std::make_shared<TestModule<float>>("sub");
  auto param2 = std::make_shared<Parameter<float>>(Tensor<float>({3, 3}));

  module_->register_parameter("weight1", param1);
  submodule->register_parameter("weight2", param2);
  module_->register_module("sub", submodule);

  auto named_params = module_->named_parameters();
  EXPECT_EQ(named_params.size(), 2);
  EXPECT_NE(named_params.find("test_module.weight1"), named_params.end());
  EXPECT_NE(named_params.find("test_module.sub.weight2"), named_params.end());

  // Verify the parameters are correct
  EXPECT_EQ(named_params["test_module.weight1"], param1);
  EXPECT_EQ(named_params["test_module.sub.weight2"], param2);
}

TEST_F(ModuleTest, NamedParametersEmpty) {
  auto named_params = module_->named_parameters();
  EXPECT_EQ(named_params.size(), 0);
}

TEST_F(ModuleTest, ParameterCount) {
  auto param1 =
      std::make_shared<Parameter<float>>(Tensor<float>({2, 2})); // 4 elements
  auto param2 =
      std::make_shared<Parameter<float>>(Tensor<float>({3, 3})); // 9 elements

  module_->register_parameter("weight1", param1);
  module_->register_parameter("weight2", param2);

  EXPECT_EQ(module_->parameter_count(), 13); // 4 + 9
}

TEST_F(ModuleTest, ParameterCountEmpty) {
  EXPECT_EQ(module_->parameter_count(), 0);
}

TEST_F(ModuleTest, ParameterCountWithSubmodules) {
  auto param1 =
      std::make_shared<Parameter<float>>(Tensor<float>({2, 2})); // 4 elements
  auto submodule = std::make_shared<TestModule<float>>("sub");
  auto param2 =
      std::make_shared<Parameter<float>>(Tensor<float>({3, 3})); // 9 elements

  module_->register_parameter("weight1", param1);
  submodule->register_parameter("weight2", param2);
  module_->register_module("sub", submodule);

  EXPECT_EQ(module_->parameter_count(), 13); // 4 + 9
}

TEST_F(ModuleTest, MemoryUsage) {
  auto param = std::make_shared<Parameter<float>>(Tensor<float>({2, 2}));
  module_->register_parameter("weight", param);

  size_t expected_usage = param->data().memory_usage();
  EXPECT_EQ(module_->memory_usage(), expected_usage);
}

TEST_F(ModuleTest, MemoryUsageEmpty) { EXPECT_EQ(module_->memory_usage(), 0); }

TEST_F(ModuleTest, MemoryUsageWithSubmodules) {
  auto param1 = std::make_shared<Parameter<float>>(Tensor<float>({2, 2}));
  auto submodule = std::make_shared<TestModule<float>>("sub");
  auto param2 = std::make_shared<Parameter<float>>(Tensor<float>({3, 3}));

  module_->register_parameter("weight1", param1);
  submodule->register_parameter("weight2", param2);
  module_->register_module("sub", submodule);

  size_t expected_usage =
      param1->data().memory_usage() + param2->data().memory_usage();
  EXPECT_EQ(module_->memory_usage(), expected_usage);
}

TEST_F(ModuleTest, DeviceTransferWithParameters) {
  if (!cuda_available_) {
    GTEST_SKIP() << "CUDA not available";
  }

  auto param = std::make_shared<Parameter<float>>(Tensor<float>({2, 2}));
  module_->register_parameter("weight", param);

  // Initially on CPU
  EXPECT_TRUE(module_->device().is_cpu());
  EXPECT_TRUE(param->data().device().is_cpu());

  module_->to(Device::cuda(0));

  // After transfer, both module and parameter should be on CUDA
  EXPECT_TRUE(module_->device().is_cuda());
  EXPECT_EQ(module_->device().rank, 0);
  EXPECT_TRUE(param->data().device().is_cuda());
  EXPECT_EQ(param->data().device().rank, 0);
}

TEST_F(ModuleTest, DeviceTransferWithSubmodules) {
  if (!cuda_available_) {
    GTEST_SKIP() << "CUDA not available";
  }

  auto submodule = std::make_shared<TestModule<float>>("sub");
  auto param = std::make_shared<Parameter<float>>(Tensor<float>({2, 2}));
  submodule->register_parameter("weight", param);
  module_->register_module("sub", submodule);

  // Initially on CPU
  EXPECT_TRUE(module_->device().is_cpu());
  EXPECT_TRUE(submodule->device().is_cpu());
  EXPECT_TRUE(param->data().device().is_cpu());

  module_->to(Device::cuda(0));

  // After transfer, module, submodule, and parameter should be on CUDA
  EXPECT_TRUE(module_->device().is_cuda());
  EXPECT_EQ(module_->device().rank, 0);
  EXPECT_TRUE(submodule->device().is_cuda());
  EXPECT_EQ(submodule->device().rank, 0);
  EXPECT_TRUE(param->data().device().is_cuda());
  EXPECT_EQ(param->data().device().rank, 0);
}

TEST_F(ModuleTest, DeviceTransferCudaToCpu) {
  if (!cuda_available_) {
    GTEST_SKIP() << "CUDA not available";
  }

  auto param = std::make_shared<Parameter<float>>(Tensor<float>({2, 2}));
  module_->register_parameter("weight", param);

  // Move to CUDA first
  module_->to(Device::cuda(0));
  EXPECT_TRUE(module_->device().is_cuda());
  EXPECT_TRUE(param->data().device().is_cuda());

  // Move back to CPU
  module_->to(Device::cpu());
  EXPECT_TRUE(module_->device().is_cpu());
  EXPECT_TRUE(param->data().device().is_cpu());
}

TEST_F(ModuleTest, OstreamOperator) {
  auto param = std::make_shared<Parameter<float>>(Tensor<float>({2, 3}));
  module_->register_parameter("weight", param);

  std::stringstream ss;
  ss << *module_;
  std::string output = ss.str();

  EXPECT_TRUE(output.find("test_module(") != std::string::npos);
  EXPECT_TRUE(output.find("Parameters:") != std::string::npos);
  EXPECT_TRUE(output.find("Total parameters:") != std::string::npos);
  EXPECT_TRUE(output.find("Memory usage:") != std::string::npos);
  EXPECT_TRUE(output.find("Device:") != std::string::npos);
}

TEST_F(ModuleTest, OstreamOperatorEmpty) {
  std::stringstream ss;
  ss << *module_;
  std::string output = ss.str();

  EXPECT_TRUE(output.find("test_module(") != std::string::npos);
  EXPECT_TRUE(output.find("Total parameters: 0") != std::string::npos);
  EXPECT_TRUE(output.find("Memory usage: 0") != std::string::npos);
  EXPECT_TRUE(output.find("Device: cpu") != std::string::npos);
}

TEST_F(ModuleTest, OstreamOperatorWithSubmodules) {
  auto submodule = std::make_shared<TestModule<float>>("sub");
  auto param = std::make_shared<Parameter<float>>(Tensor<float>({2, 2}));
  submodule->register_parameter("weight", param);
  module_->register_module("sub", submodule);

  std::stringstream ss;
  ss << *module_;
  std::string output = ss.str();

  EXPECT_TRUE(output.find("test_module(") != std::string::npos);
  EXPECT_TRUE(output.find("sub") != std::string::npos);
  EXPECT_TRUE(output.find("Total parameters: 4") != std::string::npos);
}

// Template tests
template <typename T> class TypedModuleTest : public ::testing::Test {
protected:
  void SetUp() override {
    module_ = std::make_unique<TestModule<T>>("typed_module");
  }

  std::unique_ptr<TestModule<T>> module_;
};

using TestedTypes = ::testing::Types<float, double, int32_t>;
TYPED_TEST_SUITE(TypedModuleTest, TestedTypes);

TYPED_TEST(TypedModuleTest, TypeSpecificOperations) {
  using TensorType = Tensor<TypeParam>;
  using ParameterType = Parameter<TypeParam>;

  auto param = std::make_shared<ParameterType>(TensorType({2, 2}));
  this->module_->register_parameter("weight", param);

  EXPECT_EQ(this->module_->parameter_count(), 4);
  EXPECT_EQ(this->module_->memory_usage(), 4 * sizeof(TypeParam));

  // Test forward pass
  TensorType input({2, 2});
  TensorType output = this->module_->forward(input);
  EXPECT_EQ(output.shape(), input.shape());
}

// Integration tests
TEST_F(ModuleTest, ComplexHierarchy) {
  // Create a complex module hierarchy
  auto submodule1 = std::make_shared<TestModule<float>>("sub1");
  auto submodule2 = std::make_shared<TestModule<float>>("sub2");
  auto sub_submodule = std::make_shared<TestModule<float>>("sub_sub");

  auto param1 =
      std::make_shared<Parameter<float>>(Tensor<float>({2, 2})); // 4 elements
  auto param2 =
      std::make_shared<Parameter<float>>(Tensor<float>({3, 3})); // 9 elements
  auto param3 =
      std::make_shared<Parameter<float>>(Tensor<float>({1, 5})); // 5 elements

  module_->register_parameter("main_weight", param1);
  submodule1->register_parameter("sub1_weight", param2);
  sub_submodule->register_parameter("sub_sub_weight", param3);

  submodule1->register_module("sub_sub", sub_submodule);
  module_->register_module("sub1", submodule1);
  module_->register_module("sub2", submodule2);

  // Test parameter count
  EXPECT_EQ(module_->parameter_count(), 18); // 4 + 9 + 5

  // Test named parameters
  auto named_params = module_->named_parameters();
  EXPECT_EQ(named_params.size(), 3);
  EXPECT_NE(named_params.find("test_module.main_weight"), named_params.end());
  EXPECT_NE(named_params.find("test_module.sub1.sub1_weight"),
            named_params.end());
  EXPECT_NE(named_params.find("test_module.sub1.sub_sub.sub_sub_weight"),
            named_params.end());

  // Test device transfer
  if (cuda_available_) {
    module_->to(Device::cuda(0));
    EXPECT_TRUE(module_->device().is_cuda());
    EXPECT_TRUE(submodule1->device().is_cuda());
    EXPECT_TRUE(submodule2->device().is_cuda());
    EXPECT_TRUE(sub_submodule->device().is_cuda());
    EXPECT_TRUE(param1->data().device().is_cuda());
    EXPECT_TRUE(param2->data().device().is_cuda());
    EXPECT_TRUE(param3->data().device().is_cuda());
  }
}