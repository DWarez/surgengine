#include "core/nn/parameter.cuh"
#include "core/tensor.cuh"
#include <core/device.cuh>
#include <core/nn/module.cuh>
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
    module_ = std::make_unique<TestModule<float>>("test_module");
  }

  std::unique_ptr<TestModule<float>> module_;
};

TEST_F(ModuleTest, Constructor) {
  TestModule<float> default_module;
  EXPECT_EQ(default_module.device().type, Device::cpu().type);

  TestModule<float> named_module("custom", Device::cuda(0));
  EXPECT_EQ(named_module.device().type, Device::cuda(0).type);
  EXPECT_EQ(named_module.name(), "custom");
}

TEST_F(ModuleTest, OperatorCall) {
  Tensor<float> input({2, 3});
  Tensor<float> output = (*module_)(input);
  // Output should be same as input for our test implementation
}

TEST_F(ModuleTest, DeviceGetter) {
  EXPECT_EQ(module_->device().type, Device::cpu().type);
}

TEST_F(ModuleTest, ToMethod) {
  Device cuda_device = Device::cuda(0);
  Module<float> &result = module_->to(cuda_device);

  EXPECT_EQ(module_->device().type, cuda_device.type);
  EXPECT_EQ(&result, module_.get());
}

TEST_F(ModuleTest, CudaMethod) {
  Module<float> &result = module_->cuda(1);

  EXPECT_EQ(module_->device().type, Device::cuda(1).type);
  EXPECT_EQ(&result, module_.get());
}

TEST_F(ModuleTest, CpuMethod) {
  module_->cuda(0);
  Module<float> &result = module_->cpu();

  EXPECT_EQ(module_->device().type, Device::cpu().type);
  EXPECT_EQ(&result, module_.get());
}

TEST_F(ModuleTest, NameGetterSetter) {
  EXPECT_EQ(module_->name(), "test_module");

  module_->set_name("new_name");
  EXPECT_EQ(module_->name(), "new_name");
}

TEST_F(ModuleTest, RegisterGetParameter) {
  auto param = std::make_shared<Parameter<float>>(Tensor<float>({2, 2}));
  module_->register_parameter("weight", param);

  auto retrieved = module_->get_parameter("weight");
  EXPECT_EQ(retrieved, param);

  auto null_param = module_->get_parameter("nonexistent");
  EXPECT_EQ(null_param, nullptr);
}

TEST_F(ModuleTest, RegisterGetSubmodule) {
  auto submodule = std::make_shared<TestModule<float>>("sub");
  module_->register_module("sub", submodule);

  auto retrieved = module_->get_submodule("sub");
  EXPECT_EQ(retrieved, submodule);

  auto null_module = module_->get_submodule("nonexistent");
  EXPECT_EQ(null_module, nullptr);
}

TEST_F(ModuleTest, ParametersMethod) {
  auto param1 = std::make_shared<Parameter<float>>(Tensor<float>({2, 2}));
  auto param2 = std::make_shared<Parameter<float>>(Tensor<float>({3, 3}));

  module_->register_parameter("weight1", param1);
  module_->register_parameter("weight2", param2);

  auto all_params = module_->parameters();
  EXPECT_EQ(all_params.size(), 2);
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
}

TEST_F(ModuleTest, ParameterCount) {
  auto param1 = std::make_shared<Parameter<float>>(Tensor<float>({2, 2})); // 4
  auto param2 = std::make_shared<Parameter<float>>(Tensor<float>({3, 3})); // 9

  module_->register_parameter("weight1", param1);
  module_->register_parameter("weight2", param2);

  EXPECT_EQ(module_->parameter_count(), 13);
}

TEST_F(ModuleTest, MemoryUsage) {
  auto param = std::make_shared<Parameter<float>>(Tensor<float>({2, 2}));
  module_->register_parameter("weight", param);

  size_t expected_usage = param->data().memory_usage();
  EXPECT_EQ(module_->memory_usage(), expected_usage);
}

TEST_F(ModuleTest, DeviceTransferWithParameters) {
  auto param = std::make_shared<Parameter<float>>(Tensor<float>({2, 2}));
  module_->register_parameter("weight", param);

  module_->to(Device::cuda(0));
  EXPECT_EQ(module_->device().type, Device::cuda(0).type);
}

TEST_F(ModuleTest, DeviceTransferWithSubmodules) {
  auto submodule = std::make_shared<TestModule<float>>("sub");
  module_->register_module("sub", submodule);

  module_->to(Device::cuda(0));
  EXPECT_EQ(module_->device().type, Device::cuda(0).type);
  EXPECT_EQ(submodule->device().type, Device::cuda(0).type);
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