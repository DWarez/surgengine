project('surgengine', ['cpp', 'cuda'],
        version : '1.0.0',
        default_options : ['cpp_std=c++17'])

add_project_arguments('-Wno-reorder', language: 'cpp')

json_dep = dependency('nlohmann_json', required: true)
gtest_dep = dependency('gtest', required: true)
gtest_main_dep = dependency('gtest_main', required: true)
cublas_dep = dependency('cublas', required: true)
cblas_dep = dependency('cblas', required: true)

# Find CUDA toolkit
cuda_dep = dependency('cuda', version : '>=10.0', required : true)
cuda_args = ['-gencode', 'arch=compute_89,code=sm_89']

# Find Boost
boost_dep = dependency('boost', modules: ['system', 'filesystem'], required: true)

# Add subdirectory for source files
subdir('src')
subdir('src/server')
subdir('src/kernels')
subdir('tests')