tensor_inc = include_directories('.')


cpp_sources = files('main.cpp')
cuda_sources = files('kernels/fill_kernel.cu', 'kernels/fill_kernel.hpp')

tensor_lib = static_library('tensor',
    sources: cuda_sources,  # Include the CUDA sources
    dependencies: [cuda_dep],
    cuda_args: ['-arch=sm_86'],
    include_directories: tensor_inc,
    install: false
)

tensor_dep = declare_dependency(
    link_with: tensor_lib,
    include_directories: tensor_inc
)

# Create executable
executable('surgengine',
           [cpp_sources],
           dependencies : [tensor_dep, cuda_dep, boost_dep, json_dep],
           install : true)