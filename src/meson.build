tensor_inc = include_directories('.')


cpp_sources = files('main.cpp')
cuda_sources = files('kernels/fill_kernel.cu', 'kernels/fill_kernel.hpp')

tensor_lib = static_library('tensor',
    sources: cuda_sources,
    dependencies: [cuda_dep],
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
           dependencies : [tensor_dep, cuda_dep, boost_dep, json_dep],
           cuda_args: cuda_args,
           install : true)