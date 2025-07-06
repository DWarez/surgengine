#pragma once

#include <cblas.h>
#include <core/device.cuh>
#include <core/nn/module.cuh>
#include <core/nn/parameter.cuh>
#include <core/tensor.cuh>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <type_traits>
#include <utils/cblas_utils.hpp>
#include <utils/cuda_utils.cuh>
#include <vector>

namespace surgengine {
namespace nn {
template <typename T> class LinearLayer : public Module<T> {
private:
  int in_features_;
  int out_features_;
  bool use_bias_;
  std::shared_ptr<Parameter<T>> weights_;
  std::shared_ptr<Parameter<T>> bias_;
  static std::mutex cublas_mutex_;

  static cublasHandle_t cublas_handle_;
  static bool is_cublas_initialized_;
  static int current_rank_;

  void ensure_cublas_initalized() {
    if (this->device_.is_cuda()) {
      // Todo: I'm unsure if this is required and/or has impact on performance
      std::lock_guard<std::mutex> lock(cublas_mutex_);
      int device_rank = this->device_.rank;

      if (current_rank_ != device_rank) {
        if (is_cublas_initialized_) {
          cublasDestroy(cublas_handle_);
        }
        CUDA_CHECK(cudaSetDevice(device_rank));
        cublasCreate(&cublas_handle_);
        is_cublas_initialized_ = true;
        current_rank_ = device_rank;
      }
    }
  }

  cudaDataType_t get_cuda_data_type() const {
    if constexpr (std::is_same_v<T, float>)
      return CUDA_R_32F;
    if constexpr (std::is_same_v<T, double>)
      return CUDA_R_64F;
    if constexpr (std::is_same_v<T, __half>)
      return CUDA_R_16F;
    throw std::runtime_error("Unsupported data type for cuBLAS");
  }

  void _cuda_forward(T *output_data, cublasHandle_t cublas_handle,
                     int batch_size, int in_features, int out_features_,
                     const T &alpha, const T &beta, const T *input_data,
                     const T *weights_data) {
    ensure_cublas_initalized();
    cublas_gemm_dispatch(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                         out_features_, batch_size, in_features_, &alpha,
                         weights_data, out_features_, input_data, in_features_,
                         &beta, output_data, out_features_);
    if (use_bias_) {
      const T *bias_data = bias_->data().data();
      for (int b = 0; b < batch_size; ++b) {
        cublas_axpy_dispatch(cublas_handle_, out_features_, &alpha, bias_data,
                             1, output_data + b * out_features_, 1);
      }
    }
  }

  void _cpu_forward(T *output_data, int batch_size, int in_features,
                    int out_features_, const T &alpha, const T &beta,
                    const T *input_data, const T *weights_data) {
    cblas_gemm_dispatch(CblasRowMajor, CblasNoTrans, CblasTrans, batch_size,
                        out_features_, in_features_, alpha, input_data,
                        in_features_, weights_data, in_features_, beta,
                        output_data, out_features_);

    if (use_bias_) {
      const T *bias_data = bias_->data().data();
      for (int b = 0; b < batch_size; ++b) {
        cblas_axpy_dispatch(out_features_, alpha, bias_data, 1,
                            output_data + b * out_features_, 1);
      }
    }
  }

  Tensor<T> _forward(const Tensor<T> &input) {
    int batch_size = input.shape()[0];
    std::vector<int> output_shape =
        batch_size > 1 ? std::vector<int>{batch_size, out_features_}
                       : std::vector<int>{out_features_};
    Tensor<T> output(output_shape, this->device_);

    const T *input_data = input.data();
    const T alpha = static_cast<T>(1.0);
    const T beta = static_cast<T>(0.0);
    const T *weights_data = weights_->data().data();
    T *output_data = output.data();

    if (this->device_.is_cuda())
      _cuda_forward(output_data, cublas_handle_, batch_size, in_features_,
                    out_features_, alpha, beta, input_data, weights_data);
    else
      _cpu_forward(output_data, batch_size, in_features_, out_features_, alpha,
                   beta, input_data, weights_data);
    return output;
  }

public:
  LinearLayer(int in_features, int out_features, bool use_bias = true,
              const std::string &name = "LinearLayer",
              const Device &device = Device::cpu())
      : Module<T>(name, device), in_features_(in_features),
        out_features_(out_features), use_bias_(use_bias) {
    std::vector<int> weight_shape = {out_features_, in_features_};
    weights_ = std::make_shared<Parameter<T>>(Tensor<T>(weight_shape, device));
    this->register_parameter("weight", weights_);

    if (use_bias_) {
      std::vector<int> bias_shape = {out_features_};
      bias_ = std::make_shared<Parameter<T>>(Tensor<T>(bias_shape, device));
      this->register_parameter("bias", bias_);
    }
  }

  Tensor<T> forward(const Tensor<T> &input) override {
    auto input_shape = input.shape();

    if (input_shape.empty() || input_shape.size() > 2) {
      throw std::runtime_error("Input must be 1D or 2D tensor");
    }
    int actual_in_features = 0;

    if (input_shape.size() == 1) {
      actual_in_features = input_shape[0];
    } else if (input_shape.size() == 2) {
      actual_in_features = input_shape[1];
    }
    if (actual_in_features != in_features_) {
      throw std::runtime_error("Input feature dimension (" +
                               std::to_string(actual_in_features) +
                               ") doesn't match layer input size (" +
                               std::to_string(in_features_) + ")");
    }
    Tensor<T> reshaped_input = input;
    if (input_shape.size() == 1) {
      reshaped_input = input.view({1, in_features_});
    }
    Tensor<T> output = _forward(reshaped_input);
    if (input_shape.size() == 1) {
      output = output.view({out_features_});
    }
    return output;
  }

  int in_features() const { return in_features_; }
  int out_features() const { return out_features_; }
  bool has_bias() const { return use_bias_; }

  static void cleanup_cublas() {
    if (is_cublas_initialized_) {
      cublasDestroy(cublas_handle_);
      is_cublas_initialized_ = false;
    }
  }
};

template <typename T> cublasHandle_t LinearLayer<T>::cublas_handle_ = nullptr;
template <typename T> bool LinearLayer<T>::is_cublas_initialized_ = false;
template <typename T> int LinearLayer<T>::current_rank_ = -1;
} // namespace nn
} // namespace surgengine