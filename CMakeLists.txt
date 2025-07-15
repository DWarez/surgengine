cmake_minimum_required(VERSION 3.18)
project(surgengine VERSION 1.0.0 LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler flags
add_compile_options(-Wno-reorder)

# Debug configuration
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-g -O0)
endif()

# CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES 89)

# Set LibTorch path
set(CMAKE_PREFIX_PATH "/usr/local/libtorch;${CMAKE_PREFIX_PATH}")

# Find CUDA toolkit first
find_package(CUDAToolkit 10.0 REQUIRED)

# Get PyTorch version info (with fallback)
execute_process(
    COMMAND python3 -c "from torch import version; print(version.__version__.split('+')[0])"
    OUTPUT_VARIABLE PYTORCH_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

execute_process(
    COMMAND python3 -c "from torch import version; print(version.cuda)"
    OUTPUT_VARIABLE PYTORCH_CUDA_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

# Fallback values if Python PyTorch not available
if(NOT PYTORCH_VERSION)
    set(PYTORCH_VERSION "2.1.0")  # Assume version based on LibTorch
    message(STATUS "Could not detect PyTorch version from Python, assuming: ${PYTORCH_VERSION}")
endif()

if(NOT PYTORCH_CUDA_VERSION)
    set(PYTORCH_CUDA_VERSION "12.1")  # Assume version based on downloaded LibTorch
    message(STATUS "Could not detect PyTorch CUDA version from Python, assuming: ${PYTORCH_CUDA_VERSION}")
endif()

message(STATUS "PyTorch version: ${PYTORCH_VERSION}")
message(STATUS "PyTorch CUDA version: ${PYTORCH_CUDA_VERSION}")

# PyTorch NVTX headers workaround for CUDA 12.0+ and PyTorch 2.5.0+
if(
    PYTORCH_VERSION VERSION_GREATER_EQUAL 2.0.0 AND
    PYTORCH_CUDA_VERSION VERSION_GREATER_EQUAL 12
)
    message(STATUS "PyTorch NVTX headers workaround: Yes")
    # only do this if nvToolsExt is not defined and CUDA::nvtx3 exists
    if(NOT TARGET CUDA::nvToolsExt AND TARGET CUDA::nvtx3)
        add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
        # ensure that PyTorch is told to use NVTX3 headers
        target_compile_definitions(
            CUDA::nvToolsExt INTERFACE
            TORCH_CUDA_USE_NVTX3
        )
        target_link_libraries(CUDA::nvToolsExt INTERFACE CUDA::nvtx3)
    endif()
else()
    message(STATUS "PyTorch NVTX headers workaround: No")
endif()

# Set policy to suppress Boost warning
cmake_policy(SET CMP0167 NEW)

# Find LibTorch
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Find other required packages
find_package(Boost REQUIRED COMPONENTS system filesystem)
find_package(nlohmann_json 3.2.0 REQUIRED)
find_package(GTest REQUIRED)

# Enable testing BEFORE adding subdirectories
enable_testing()

# Add subdirectories
add_subdirectory(src)
add_subdirectory(tests)