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