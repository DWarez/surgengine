# Create individual test executables
add_executable(device_test test_device.cpp)
add_executable(tensor_test test_tensor.cu)
add_executable(parameter_test test_parameter.cu)
add_executable(module_test test_module.cu)
add_executable(linear_layer_test test_linear_layer.cu)

# Set CUDA properties for .cu files
set_target_properties(tensor_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(parameter_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(module_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(linear_layer_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Link libraries to all test executables
set(TEST_LIBRARIES
    tensor
    CUDA::cudart
    CUDA::cublas
    GTest::gtest
    GTest::gtest_main
    ${CBLAS_LIBRARIES}
    ${BLAS_LIBRARIES}
)

target_link_libraries(device_test ${TEST_LIBRARIES})
target_link_libraries(tensor_test ${TEST_LIBRARIES})
target_link_libraries(parameter_test ${TEST_LIBRARIES})
target_link_libraries(module_test ${TEST_LIBRARIES})
target_link_libraries(linear_layer_test ${TEST_LIBRARIES})

# Include directories for all tests
set(TEST_INCLUDE_DIRS
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(device_test PRIVATE ${TEST_INCLUDE_DIRS})
target_include_directories(tensor_test PRIVATE ${TEST_INCLUDE_DIRS})
target_include_directories(parameter_test PRIVATE ${TEST_INCLUDE_DIRS})
target_include_directories(module_test PRIVATE ${TEST_INCLUDE_DIRS})
target_include_directories(linear_layer_test PRIVATE ${TEST_INCLUDE_DIRS})

# Register tests with CTest - use full path to executables
add_test(NAME device_test COMMAND $<TARGET_FILE:device_test>)
add_test(NAME tensor_test COMMAND $<TARGET_FILE:tensor_test>)
add_test(NAME parameter_test COMMAND $<TARGET_FILE:parameter_test>)
add_test(NAME module_test COMMAND $<TARGET_FILE:module_test>)
add_test(NAME linear_layer_test COMMAND $<TARGET_FILE:linear_layer_test>)

# Set working directory for tests (optional, but can be helpful)
set_tests_properties(device_test tensor_test parameter_test module_test linear_layer_test
    PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)