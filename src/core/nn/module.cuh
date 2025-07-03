#pragma once
#include "core/device.hpp"
#include "core/nn/parameter.cuh"
#include <core/tensor.cuh>
#include <memory>
#include <string>
#include <unordered_map>

namespace surgengine {
namespace nn {
template <typename T> class Module {
protected:
  std::unordered_map<std::string, std::shared_ptr<Parameter<T>>> parameters_;
  std::unordered_map<std::string, std::shared_ptr<Module<T>>> modules_;
  Device device_;
  std::string name_;

public:
  Module(const std::string &name = "", const Device &device = Device::cpu())
      : name_(name), device_(device) {}

  virtual ~Module() = default;

  virtual Tensor<T> forward(const Tensor<T> &input) = 0;

  Tensor<T> operator()(const Tensor<T> &input) { return forward(input); }

  Module<T> &to(const Device &device) {
    device_ = device;

    for (auto &[name, param] : parameters_) {
      if (param->data().device() != device) {
        Tensor<T> new_tensor(param->data().shape(), device);
      }
    }
  }
};
} // namespace nn
} // namespace surgengine