#include "linear_layer.cuh"

namespace surgengine {
namespace nn {

template <typename T> std::mutex LinearLayer<T>::cublas_mutex_;

template class LinearLayer<float>;
template class LinearLayer<double>;

} // namespace nn
} // namespace surgengine