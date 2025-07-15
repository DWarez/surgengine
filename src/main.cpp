#include <iostream>
#include <torch/torch.h>

int main() {
  std::cout << "LibTorch version: " << TORCH_VERSION_MAJOR << "."
            << TORCH_VERSION_MINOR << std::endl;

  torch::Tensor tensor = torch::rand({2, 3});
  std::cout << "Random tensor:\n" << tensor << std::endl;

  if (torch::cuda::is_available()) {
    std::cout << "CUDA is available!" << std::endl;
    torch::Tensor cuda_tensor =
        torch::rand({2, 3}, torch::device(torch::kCUDA));
    std::cout << "CUDA tensor:\n" << cuda_tensor << std::endl;
  } else {
    std::cout << "CUDA is not available." << std::endl;
  }

  return 0;
}