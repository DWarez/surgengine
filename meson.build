project('surgengine', ['cpp', 'cuda'],
        version : '1.0.0',
        default_options : ['cpp_std=c++17'])


json_dep = dependency('nlohmann_json', required: true)

# Find CUDA toolkit
cuda_dep = dependency('cuda', version : '>=10.0', required : true)

# Find Boost
boost_dep = dependency('boost', modules: ['system', 'filesystem'], required: true)

# Add subdirectory for source files
subdir('src')