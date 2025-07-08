#include <cstdint>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

class TensorShape {
private:
  std::vector<int64_t> dims_;
  std::vector<int64_t> strides_;

  void compute_strides() {
    strides_.resize(dims_.size());
    if (dims_.empty())
      return;

    strides_.back() = 1;
    for (int i = dims_.size() - 2; i >= 0; --i) {
      strides_[i] = strides_[i + 1] * dims_[i + 1];
    }
  }

public:
  TensorShape() = default;

  explicit TensorShape(const std::vector<int64_t> &dimensions)
      : dims_(dimensions) {
    compute_strides();
  }

  explicit TensorShape(const std::vector<int64_t> &dimensions,
                       const std::vector<int64_t> &strides)
      : dims_(dimensions), strides_(strides) {}

  TensorShape(std::initializer_list<int64_t> dimensions) : dims_(dimensions) {
    compute_strides();
  }

  TensorShape(const TensorShape &other)
      : dims_(other.dims_), strides_(other.strides_) {}

  TensorShape(TensorShape &&other) noexcept
      : dims_(std::move(other.dims_)), strides_(std::move(other.strides_)) {}

  TensorShape &operator=(const TensorShape &other) {
    if (this != &other) {
      dims_ = other.dims_;
      strides_ = other.strides_;
    }
    return *this;
  }

  TensorShape &operator=(TensorShape &&other) noexcept {
    if (this != &other) {
      dims_ = std::move(other.dims_);
      strides_ = std::move(other.strides_);
    }
    return *this;
  }

  const std::vector<int64_t> &dims() const { return dims_; }
  const std::vector<int64_t> &strides() const { return strides_; }

  int64_t dim(int index) const {
    if (index < 0)
      index += dims_.size();
    if (index < 0 || index >= static_cast<int>(dims_.size())) {
      throw std::out_of_range("Dimension index out of range");
    }
    return dims_[index];
  }

  int64_t stride(int index) const {
    if (index < 0)
      index += strides_.size();
    if (index < 0 || index >= static_cast<int>(strides_.size())) {
      throw std::out_of_range("Stride index out of range");
    }
    return strides_[index];
  }

  int64_t numel() const {
    return dims_.empty() ? 0
                         : std::accumulate(dims_.begin(), dims_.end(), 1LL,
                                           std::multiplies<int64_t>());
  }

  int ndim() const { return static_cast<int>(dims_.size()); }

  bool empty() const { return dims_.empty(); }

  void set_dims(const std::vector<int64_t> &new_dims) {
    dims_ = new_dims;
    compute_strides();
  }

  void reshape(const std::vector<int64_t> &new_dims) {
    if (numel() != std::accumulate(new_dims.begin(), new_dims.end(), 1LL,
                                   std::multiplies<int64_t>())) {
      throw std::invalid_argument(
          "Cannot reshape: total size must remain the same");
    }
    dims_ = new_dims;
    compute_strides();
  }

  void print() const {
    std::cout << "[";
    for (size_t i = 0; i < dims_.size(); ++i) {
      if (i > 0)
        std::cout << ", ";
      std::cout << dims_[i];
    }
    std::cout << "]";
  }

  bool operator==(const TensorShape &other) const {
    return dims_ == other.dims_;
  }

  bool operator!=(const TensorShape &other) const { return !(*this == other); }

  int64_t operator[](int index) const { return dim(index); }
};