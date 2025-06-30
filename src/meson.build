cpp_sources = files('main.cpp')
cuda_sources = files('kernels/fill_kernel.cu', 'kernels/fill_kernel.hpp')

# Create executable
executable('surgengine',
           [cpp_sources, cuda_sources],
           dependencies : [cuda_dep, boost_dep, json_dep],
           install : true)