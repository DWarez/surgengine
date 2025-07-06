#pragma once

#include <core/device.cuh>
#include <core/nn/init.cuh>
#include <core/tensor.cuh>

using namespace surgengine;
using namespace surgengine::nn;

namespace surgengine {
namespace nn {
template <typename T> class Parameter;

template <typename T> class Parameter {
private:
  Tensor<T> data_;
  std::string name_;
  Device device_;

public:
  Parameter(const std::string &name = "", const Device &device = Device::cpu())
      : data_(Tensor<T>(std::vector<int>{1, 1}, device)), name_(name),
        device_(device) {}

  Parameter(const Tensor<T> &tensor, const std::string &name = "",
            const Device &device = Device::cpu())
      : data_(tensor), name_(name), device_(device) {}

  Tensor<T> &data() { return data_; }
  const Tensor<T> &data() const { return data_; }

  const std::string &name() const { return name_; }
  void set_name(const std::string &name) { name_ = name; }

  const Device &device() const { return device_; }

  Parameter<T> &uniform_(T low = -1.0, T high = 1.0) {
    init::uniform_(data_, low, high);
    return *this;
  }

  Parameter<T> &normal_(T mean = 0.0, T std = 1.0) {
    init::normal_(data_, mean, std);
    return *this;
  }

  Parameter<T> &xavier_uniform_() {
    init::xavier_uniform_(data_);
    return *this;
  }

  Parameter<T> &zero_() {
    init::zeros_(data_);
    return *this;
  }

  Parameter<T> &fill_(T value) {
    init::constant_(data_, value);
    return *this;
  }

  Parameter<T> &ones_() {
    init::ones_(data_);
    return *this;
  }
};
} // namespace nn
} // namespace surgengine