# Include directories
set(TENSOR_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR})

# Source files
set(CPP_SOURCES
    main.cpp
)

set(CUDA_SOURCES
    kernels/fill_kernel.cu
)

# Create static library
add_library(tensor STATIC ${CUDA_SOURCES})

# Set properties for the library
set_target_properties(tensor PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Link libraries to tensor
target_link_libraries(tensor
    CUDA::cudart
    CUDA::cublas
    ${CBLAS_LIBRARIES}
    ${BLAS_LIBRARIES}
)

# Include directories for tensor
target_include_directories(tensor PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/core
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels
)

# Create main executable
add_executable(surgengine ${CPP_SOURCES})

# Link libraries to main executable
target_link_libraries(surgengine
    tensor
    CUDA::cudart
    Boost::system
    Boost::filesystem
    nlohmann_json::nlohmann_json
    ${CBLAS_LIBRARIES}
    ${BLAS_LIBRARIES}
    CUDA::cublas
)

# Include directories for main executable
target_include_directories(surgengine PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Install the executable
install(TARGETS surgengine DESTINATION bin)