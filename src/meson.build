subdir('server')

cpp_sources = files('main.cpp')
cuda_sources = files('kernel.cu')

# Create executable
executable('surgengine',
           [cpp_sources, cuda_sources],
           dependencies : [cuda_dep, boost_dep, json_dep],
           install : true)