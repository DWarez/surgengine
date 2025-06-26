project('surgengine', ['cpp', 'cuda'],
        version : '1.0.0',
        default_options : ['cpp_std=c++17'])

# Find CUDA toolkit
cuda_dep = dependency('cuda', version : '>=10.0', required : true)

# Add subdirectory for source files
subdir('src')