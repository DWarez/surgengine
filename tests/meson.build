test_device_exe = executable('device_test',
    sources: [
        'test_device.cpp',
    ],
    dependencies: [
      tensor_dep,
        cuda_dep,
        gtest_dep,
        gtest_main_dep
    ],
    cuda_args: cuda_args
)

test_tensor_exe = executable('tensor_test',
    sources: [
        'test_tensor.cu',
    ],
    dependencies: [
      tensor_dep,
        cuda_dep,
        gtest_dep,
        gtest_main_dep
    ],
    cuda_args: cuda_args
)

test_parameter_exe = executable('parameter_test',
    sources: [
        'test_parameter.cu',
    ],
    dependencies: [
      tensor_dep,
        cuda_dep,
        gtest_dep,
        gtest_main_dep
    ],
    cuda_args: cuda_args
)
# Register the test
test('device_test', test_device_exe)
test('tensor_test', test_tensor_exe)
test('parameter_test', test_parameter_exe)