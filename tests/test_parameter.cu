#include <core/nn/module.cuh>
#include <gtest/gtest.h>

using namespace surgengine::nn;

class ParameterTest : public ::testing::Test {
protected:
  void SetUp() override {}
};

TEST_F(ParameterTest, DefaultConstruction) {
  Parameter<float> param;

  EXPECT_EQ(param.name(), "");
  EXPECT_TRUE(param.device().is_cpu());
  EXPECT_EQ(param.data().shape(), (std::vector<int>{1, 1}));
  EXPECT_EQ(param.data().numel(), 1);
}

TEST_F(ParameterTest, ConstructionWithName) {
  Parameter<float> param("test_param");

  EXPECT_EQ(param.name(), "test_param");
  EXPECT_TRUE(param.device().is_cpu());
}

TEST_F(ParameterTest, ConstructionWithNameAndDevice) {
  Parameter<float> param("gpu_param", Device::cuda());

  EXPECT_EQ(param.name(), "gpu_param");
  EXPECT_TRUE(param.device().is_cuda());
}

TEST_F(ParameterTest, ConstructionWithTensor) {
  FloatTensor tensor({3, 4});
  Parameter<float> param(tensor, "tensor_param");

  EXPECT_EQ(param.name(), "tensor_param");
  EXPECT_EQ(param.data().shape(), (std::vector<int>{3, 4}));
  EXPECT_EQ(param.data().numel(), 12);
}

TEST_F(ParameterTest, DataAccessors) {
  Parameter<float> param;

  FloatTensor &data_ref = param.data();
  data_ref.fill_(5.0f);

  const Parameter<float> &const_param = param;
  const FloatTensor &const_data = const_param.data();

  EXPECT_FLOAT_EQ(const_data.data()[0], 5.0f);
}

TEST_F(ParameterTest, NameMutator) {
  Parameter<float> param;

  EXPECT_EQ(param.name(), "");

  param.set_name("new_name");
  EXPECT_EQ(param.name(), "new_name");

  param.set_name("another_name");
  EXPECT_EQ(param.name(), "another_name");
}

TEST_F(ParameterTest, UniformInitialization) {
  Parameter<float> param;

  Parameter<float> &result = param.uniform_(-2.0f, 2.0f);
  EXPECT_EQ(&result, &param); // Method chaining

  const float *data = param.data().data();
  for (size_t i = 0; i < param.data().numel(); ++i) {
    EXPECT_GE(data[i], -2.0f);
    EXPECT_LE(data[i], 2.0f);
  }
}

TEST_F(ParameterTest, UniformDefaultRange) {
  Parameter<float> param;
  param.uniform_();

  const float *data = param.data().data();
  for (size_t i = 0; i < param.data().numel(); ++i) {
    EXPECT_GE(data[i], -1.0f);
    EXPECT_LE(data[i], 1.0f);
  }
}

TEST_F(ParameterTest, NormalInitialization) {
  Parameter<float> param;

  Parameter<float> &result = param.normal_(5.0f, 2.0f);
  EXPECT_EQ(&result, &param);

  EXPECT_NE(param.data().data(), nullptr);
}

TEST_F(ParameterTest, NormalDefaultParameters) {
  Parameter<float> param;
  param.normal_();

  EXPECT_NE(param.data().data(), nullptr);
}

TEST_F(ParameterTest, XavierUniformInitialization) {
  FloatTensor tensor({2, 2});
  Parameter<float> param(tensor);

  Parameter<float> &result = param.xavier_uniform_();
  EXPECT_EQ(&result, &param);

  const float *data = param.data().data();
  bool has_non_zero = false;
  for (size_t i = 0; i < param.data().numel(); ++i) {
    if (data[i] != 0.0f) {
      has_non_zero = true;
      break;
    }
  }
  EXPECT_TRUE(has_non_zero);
}

TEST_F(ParameterTest, ZeroInitialization) {
  FloatTensor tensor({3, 3});
  tensor.fill_(99.0f);
  Parameter<float> param(tensor);

  Parameter<float> &result = param.zero_();
  EXPECT_EQ(&result, &param);

  for (size_t i = 0; i < param.data().numel(); ++i) {
    EXPECT_FLOAT_EQ(param.data().data()[i], 0.0f);
  }
}

TEST_F(ParameterTest, FillInitialization) {
  Parameter<float> param;

  Parameter<float> &result = param.fill_(3.14f);
  EXPECT_EQ(&result, &param);

  for (size_t i = 0; i < param.data().numel(); ++i) {
    EXPECT_FLOAT_EQ(param.data().data()[i], 3.14f);
  }
}

TEST_F(ParameterTest, OnesInitialization) {
  Parameter<float> param;

  Parameter<float> &result = param.ones_();
  EXPECT_EQ(&result, &param);

  for (size_t i = 0; i < param.data().numel(); ++i) {
    EXPECT_FLOAT_EQ(param.data().data()[i], 1.0f);
  }
}

TEST_F(ParameterTest, MethodChaining) {
  Parameter<float> param;

  Parameter<float> &result = param.zero_().fill_(5.0f).ones_();
  EXPECT_EQ(&result, &param);

  for (size_t i = 0; i < param.data().numel(); ++i) {
    EXPECT_FLOAT_EQ(param.data().data()[i], 1.0f);
  }
}

TEST_F(ParameterTest, UniformSameRange) {
  Parameter<float> param;
  param.uniform_(2.5f, 2.5f);

  for (size_t i = 0; i < param.data().numel(); ++i) {
    EXPECT_FLOAT_EQ(param.data().data()[i], 2.5f);
  }
}

TEST_F(ParameterTest, MultipleOperations) {
  FloatTensor tensor({4, 5});
  Parameter<float> param(tensor, "multi_param", Device::cpu());

  param.zero_();
  EXPECT_FLOAT_EQ(param.data().data()[0], 0.0f);

  param.fill_(7.0f);
  EXPECT_FLOAT_EQ(param.data().data()[10], 7.0f);

  param.ones_();
  EXPECT_FLOAT_EQ(param.data().data()[19], 1.0f);

  EXPECT_EQ(param.name(), "multi_param");
}

template <typename T> class TypedParameterTest : public ParameterTest {};

using TestedTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(TypedParameterTest, TestedTypes);

TYPED_TEST(TypedParameterTest, TypeSpecificOperations) {
  Parameter<TypeParam> param;

  param.fill_(TypeParam{42});
  EXPECT_EQ(param.data().data()[0], TypeParam{42});

  param.zero_();
  EXPECT_EQ(param.data().data()[0], TypeParam{0});

  param.ones_();
  EXPECT_EQ(param.data().data()[0], TypeParam{1});
}

TYPED_TEST(TypedParameterTest, LargerTensor) {
  Tensor<TypeParam> tensor({10, 10});
  Parameter<TypeParam> param(tensor, "large_param");

  param.fill_(TypeParam{3.14});

  EXPECT_EQ(param.data().numel(), 100);
  EXPECT_EQ(param.data().data()[0], TypeParam{3.14});
  EXPECT_EQ(param.data().data()[99], TypeParam{3.14});
}

TEST_F(ParameterTest, CompleteWorkflow) {

  FloatTensor tensor = FloatTensor::zeros({6, 8});
  Parameter<float> param(tensor, "workflow_test", Device::cpu());

  param.uniform_(-1.0f, 1.0f);
  param.set_name("updated_workflow");

  EXPECT_EQ(param.name(), "updated_workflow");
  EXPECT_EQ(param.data().numel(), 48);
  EXPECT_TRUE(param.device().is_cpu());

  const float *data = param.data().data();
  for (size_t i = 0; i < param.data().numel(); ++i) {
    EXPECT_GE(data[i], -1.0f);
    EXPECT_LE(data[i], 1.0f);
  }
}