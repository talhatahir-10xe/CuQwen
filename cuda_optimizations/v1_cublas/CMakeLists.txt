cmake_minimum_required(VERSION 3.18)

# Check if gpu_arch is provided
if(NOT DEFINED gpu_arch)
    message(FATAL_ERROR 
        "\nERROR: GPU architecture is not defined!\n"
        "You must specify the target GPU architecture using -Dgpu_arch=<arch>\n\n"
        "Example usage:\n"
        "  cmake -Dgpu_arch=75 ..      # For Turing (e.g., GTX 1660, RTX 2080)\n"
        "  cmake -Dgpu_arch=80 ..      # For Ampere (e.g., A100)\n"
        "  cmake -Dgpu_arch=86 ..      # For Ampere (e.g., RTX 3080, RTX 3090)\n"
        "  cmake -Dgpu_arch=89 ..      # For Ada Lovelace (e.g., RTX 4080, RTX 4090)\n"
        "  cmake -Dgpu_arch=90 ..      # For Hopper (e.g., H100)\n"
    )
endif()

# Set the CMAKE_CUDA_ARCHITECTURES
set(CMAKE_CUDA_ARCHITECTURES ${gpu_arch})

project(cuqwen LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# Compile flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math -Xcompiler -Wall")

# Find PCRE2 library (required by tokenizer)
find_package(PkgConfig REQUIRED)
pkg_check_modules(PCRE2 REQUIRED libpcre2-8)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${PCRE2_INCLUDE_DIRS}
)

# Source compilation files
set(SOURCES
    src/main.cu
    src/load_weights.cpp
    src/kernels.cu
    src/model.cu
    src/tokenizer.cpp
)

# Build binary executable
add_executable(cuqwen ${SOURCES})

# Target CUDA runtime, cuBLAS, and PCRE2 libraries
target_link_libraries(cuqwen PRIVATE 
    CUDA::cudart 
    CUDA::cublas 
    ${PCRE2_LIBRARIES}
)