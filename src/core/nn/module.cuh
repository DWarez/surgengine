#pragma once
#include "core/nn/parameter.cuh"
#include <core/device.cuh>
#include <core/tensor.cuh>
#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>

namespace surgengine {
namespace nn {
template <typename T> class Module {
protected:
  Device device_;
  std::string name_;
  std::unordered_map<std::string, std::shared_ptr<Parameter<T>>> parameters_;
  std::unordered_map<std::string, std::shared_ptr<Module<T>>> submodules_;

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
        param->data().to(device);
      }
    }

    for (auto &[name, module] : submodules_) {
      module->to(device);
    }
    return *this;
  }

  Module<T> &cuda(int rank = 0) { return to(Device::cuda(rank)); }
  Module<T> &cpu() { return to(Device::cpu()); }

  const Device &device() const { return device_; }

  void register_parameter(const std::string &name,
                          std::shared_ptr<Parameter<T>> param) {
    param->set_name(name);
    parameters_[name] = param;
  }

  std::shared_ptr<Parameter<T>> get_parameter(const std::string &name) {
    auto it = parameters_.find(name);
    return (it != parameters_.end()) ? it->second : nullptr;
  }

  void register_module(const std::string &name,
                       std::shared_ptr<Module<T>> module) {
    submodules_[name] = module;
  }

  std::shared_ptr<Module<T>> get_submodule(const std::string &name) {
    auto it = submodules_.find(name);
    return (it != submodules_.end()) ? it->second : nullptr;
  }

  std::vector<std::shared_ptr<Parameter<T>>> parameters() const {
    std::vector<std::shared_ptr<Parameter<T>>> all_params;
    for (const auto &[name, param] : parameters_) {
      all_params.push_back(param);
    }
    for (const auto &[name, module] : submodules_) {
      auto sub_params = module->parameters();
      all_params.insert(all_params.end(), sub_params.begin(), sub_params.end());
    }

    return all_params;
  }

  std::unordered_map<std::string, std::shared_ptr<Parameter<T>>>
  named_parameters() const {
    std::unordered_map<std::string, std::shared_ptr<Parameter<T>>> all_params;

    for (const auto &[param_name, param] : parameters_) {
      std::string full_name =
          name_.empty() ? param_name : name_ + "." + param_name;
      all_params[full_name] = param;
    }

    for (const auto &[sub_name, sub_module] : submodules_) {
      auto sub_params = sub_module->named_parameters();
      for (const auto &[sub_param_name, sub_param] : sub_params) {
        std::string full_name;
        full_name = name_ + "." + sub_param_name;
        all_params[full_name] = sub_param;
      }
    }

    return all_params;
  }

  size_t parameter_count() const {
    size_t count = 0;
    for (const auto &param : parameters()) {
      count += param->data().numel();
    }
    return count;
  }

  size_t memory_usage() const {
    size_t usage = 0;
    for (const auto &param : parameters()) {
      usage += param->data().memory_usage();
    }
    return usage;
  }

  const std::string &name() const { return name_; }
  void set_name(const std::string &name) { name_ = name; }

  friend std::ostream &operator<<(std::ostream &os, const Module<T> &module) {
    os << module.name() << "(\n";

    auto named_params = module.named_parameters();
    if (!named_params.empty()) {
      os << "  Parameters:\n";
      for (const auto &[name, param] : named_params) {
        os << "    " << name << ": [";
        for (size_t i = 0; i < param->data().shape().size(); ++i) {
          os << param->data().shape()[i];
          if (i < param->data().shape().size() - 1)
            os << ", ";
        }
        os << "] (" << param->data().numel() << " elements)\n";
      }
    }

    os << "  Total parameters: " << module.parameter_count() << "\n";
    os << "  Memory usage: " << module.memory_usage() << " bytes\n";
    os << "  Device: "
       << (module.device().is_cuda()
               ? "cuda:" + std::to_string(module.device().rank)
               : "cpu")
       << "\n";
    os << ")";

    return os;
  }
};
} // namespace nn
} // namespace surgengine