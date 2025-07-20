# Collect all source files
file(GLOB_RECURSE TENSOR_SOURCES
    "core/*.cpp"
    "core/*.cu"
    "core/nn/*.cpp"
    "core/nn/*.cu"
    "kernels/*.cu"
    "utils/*.cpp"
    "utils/*.cu"
)

# Create the tensor library
add_library(tensor ${TENSOR_SOURCES})

# Set CUDA properties
set_target_properties(tensor PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Include directories
target_include_directories(tensor PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/core
    ${CMAKE_CURRENT_SOURCE_DIR}/core/nn
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels
    ${CMAKE_CURRENT_SOURCE_DIR}/utils
)

# Link required libraries
target_link_libraries(tensor
    CUDA::cudart
    CUDA::cublas
    ${CBLAS_LIBRARIES}
    ${BLAS_LIBRARIES}
    nlohmann_json::nlohmann_json
)

# Create the main executable (if needed)
add_executable(surgengine main.cpp)
target_link_libraries(surgengine tensor)
set_target_properties(surgengine PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)