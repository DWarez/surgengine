test_exe = executable('tensor_tests',
    sources: [
        'test_storage.cu'
    ],
    dependencies: [
      tensor_dep,
        cuda_dep,
        gtest_dep,
        gtest_main_dep
    ],
    cuda_args: ['-arch=sm_86']
)

# Register the test
test('tensor_tests', test_exe)
