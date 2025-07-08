tensor_inc = include_directories('.')


cpp_sources = files('main.cpp', 'core/nn/linear_layer.cpp')
cuda_sources = files('kernels/fill_kernel.cu', 'kernels/fill_kernel.hpp', 'kernels/init_kernels.cu', 'kernels/init_kernels.hpp')

tensor_lib = static_library('tensor',
    sources: [cpp_sources, cuda_sources],
    dependencies: [cuda_dep, boost_dep, json_dep, cblas_dep, cublas_dep, openmp_dep],
    include_directories: tensor_inc,
    cuda_args: cuda_args,
    install: false
)

tensor_dep = declare_dependency(
    link_with: tensor_lib,
    include_directories: tensor_inc
)

# Create executable
executable('surgengine',
           [cpp_sources],
           dependencies : [tensor_dep, cuda_dep, boost_dep, json_dep, cblas_dep, cublas_dep, openmp_dep],
           cuda_args: cuda_args,
           install : true)