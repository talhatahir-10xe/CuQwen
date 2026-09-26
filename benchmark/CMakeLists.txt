cmake_minimum_required(VERSION 3.18)

# 1. Compulsory Input: -Dmodel=<0.5b|1.5b|3b|7b>
if(NOT DEFINED model)
    message(FATAL_ERROR 
        "\n[!] Error: 'model' variable is compulsory.\n"
        "Please specify model size using -Dmodel=<size>.\n"
        "Available options: 0.5b, 1.5b, 3b, 7b\n"
        "Example:\n"
        "  cmake -Dmodel=0.5b -Dgpu_arch=75 ..\n"
    )
endif()

string(TOLOWER "${model}" MODEL_SIZE)

if(MODEL_SIZE STREQUAL "0.5b")
    add_compile_definitions(QWEN_MODEL_0_5B)
    set(MODEL_NAME_STR "0.5B")
elseif(MODEL_SIZE STREQUAL "1.5b")
    add_compile_definitions(QWEN_MODEL_1_5B)
    set(MODEL_NAME_STR "1.5B")
elseif(MODEL_SIZE STREQUAL "3b")
    add_compile_definitions(QWEN_MODEL_3B)
    set(MODEL_NAME_STR "3B")
elseif(MODEL_SIZE STREQUAL "7b")
    add_compile_definitions(QWEN_MODEL_7B)
    set(MODEL_NAME_STR "7B")
else()
    message(FATAL_ERROR "Invalid model option '${model}'. Options are: 0.5b, 1.5b, 3b, 7b")
endif()

# 1b. Optional Input: -Dquant=<fp16|int8>  (default: fp16). Must match the mode
#     used by export_weights.py, since the benchmark shares ../src and ../include.
if(NOT DEFINED quant)
    set(quant "fp16")
endif()

string(TOLOWER "${quant}" QUANT_MODE)

if(QUANT_MODE STREQUAL "int8")
    add_compile_definitions(QUANT_INT8)
elseif(QUANT_MODE STREQUAL "int4")
    add_compile_definitions(QUANT_INT4)
elseif(NOT QUANT_MODE STREQUAL "fp16")
    message(FATAL_ERROR "Invalid quant option '${quant}'. Options are: fp16, int8, int4")
endif()

# 2. Check if gpu_arch is provided
if(NOT DEFINED gpu_arch)
    message(FATAL_ERROR 
        "\nERROR: GPU architecture is not defined!\n"
        "You must specify the target GPU architecture using -Dgpu_arch=<arch>\n\n"
        "Example usage:\n"
        "  cmake -Dmodel=${model} -Dgpu_arch=75 ..      # For Turing (e.g., GTX 1660, RTX 2080)\n"
        "  cmake -Dmodel=${model} -Dgpu_arch=80 ..      # For Ampere (e.g., A100)\n"
        "  cmake -Dmodel=${model} -Dgpu_arch=86 ..      # For Ampere (e.g., RTX 3080, RTX 3090)\n"
        "  cmake -Dmodel=${model} -Dgpu_arch=89 ..      # For Ada Lovelace (e.g., RTX 4080, RTX 4090)\n"
        "  cmake -Dmodel=${model} -Dgpu_arch=90 ..      # For Hopper (e.g., H100)\n"
    )
endif()

# Set CMAKE_CUDA_ARCHITECTURES
set(CMAKE_CUDA_ARCHITECTURES ${gpu_arch})

project(cuqwen_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# Compile flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math -Xcompiler -Wall")

# Find PCRE2 library
find_package(PkgConfig REQUIRED)
pkg_check_modules(PCRE2 REQUIRED libpcre2-8)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${PCRE2_INCLUDE_DIRS}
)

# Source files
set(BENCHMARK_SOURCES
    cuqwen_benchmark.cu
    ../src/load_weights.cpp
    ../src/kernels.cu
    ../src/model.cu
    ../src/tokenizer.cpp
)

# Build benchmark executable
add_executable(cuqwen_benchmark ${BENCHMARK_SOURCES})

# Target CUDA runtime, and PCRE2 libraries
target_link_libraries(cuqwen_benchmark PRIVATE 
    CUDA::cudart
    ${PCRE2_LIBRARIES}
)