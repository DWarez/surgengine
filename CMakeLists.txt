cmake_minimum_required(VERSION 3.18)
project(surgengine VERSION 1.0.0 LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# IMPORTANT: Set debug flags BEFORE any add_compile_options
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "Building in Debug mode with debug symbols")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")
    set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -O0")
    add_compile_definitions(DEBUG)
else()
    message(STATUS "Building in Release mode")
endif()

# Compiler flags (remove the -Wno-reorder for now to test)
# add_compile_options(-Wno-reorder)

# CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES 89)

# Find required packages
find_package(PkgConfig REQUIRED)
find_package(CUDAToolkit 10.0 REQUIRED)
find_package(Boost REQUIRED COMPONENTS system filesystem)

# Find nlohmann_json
find_package(nlohmann_json 3.2.0 REQUIRED)

# Find GTest
find_package(GTest REQUIRED)

# Find CBLAS and CUBLAS
find_package(BLAS REQUIRED)
find_library(CBLAS_LIBRARIES cblas)
if(NOT CBLAS_LIBRARIES)
    message(FATAL_ERROR "CBLAS library not found")
endif()

# Enable testing BEFORE adding subdirectories
enable_testing()

# Add subdirectories
add_subdirectory(src)
add_subdirectory(tests)