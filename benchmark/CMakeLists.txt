cmake_minimum_required(VERSION 3.18)

# 1. Compulsory Input: -Dmodel=<qwen2.5|qwen3>   (model family / architecture)
if(NOT DEFINED model)
    message(FATAL_ERROR
        "\n[!] Error: 'model' variable is compulsory.\n"
        "Please specify the model family using -Dmodel=<family>.\n"
        "Available options: qwen2.5, qwen3\n"
        "Example:\n"
        "  cmake -Dmodel=qwen3 -Dmodelsize=0.6b -Dquant=int4 -Dgpu_arch=75 ..\n"
    )
endif()

string(TOLOWER "${model}" MODEL_FAMILY)

if(MODEL_FAMILY STREQUAL "qwen2.5")
    add_compile_definitions(ARCH_QWEN25)
    set(ARCH_NAME_STR "Qwen2.5")
    set(VALID_SIZES "0.5b" "1.5b" "3b" "7b" "14b")
elseif(MODEL_FAMILY STREQUAL "qwen3")
    add_compile_definitions(ARCH_QWEN3)
    set(ARCH_NAME_STR "Qwen3")
    set(VALID_SIZES "0.6b" "1.7b" "4b" "8b" "14b")
else()
    message(FATAL_ERROR "Invalid model family '${model}'. Options are: qwen2.5, qwen3")
endif()

# 1b. Compulsory Input: -Dmodelsize=<size>   (size within the chosen family)
if(NOT DEFINED modelsize)
    message(FATAL_ERROR
        "\n[!] Error: 'modelsize' variable is compulsory.\n"
        "Please specify the model size using -Dmodelsize=<size>.\n"
        "Available options for ${ARCH_NAME_STR}: ${VALID_SIZES}\n"
        "Example:\n"
        "  cmake -Dmodel=qwen3 -Dmodelsize=0.6b -Dquant=int4 -Dgpu_arch=75 ..\n"
    )
endif()

string(TOLOWER "${modelsize}" MODEL_SIZE)

list(FIND VALID_SIZES "${MODEL_SIZE}" _size_idx)
if(_size_idx EQUAL -1)
    message(FATAL_ERROR
        "Invalid modelsize '${modelsize}' for family '${MODEL_FAMILY}'. Options are: ${VALID_SIZES}")
endif()

if(MODEL_SIZE STREQUAL "0.5b")
    add_compile_definitions(QWEN_SIZE_0_5B)
    set(MODEL_NAME_STR "0.5B")
elseif(MODEL_SIZE STREQUAL "0.6b")
    add_compile_definitions(QWEN_SIZE_0_6B)
    set(MODEL_NAME_STR "0.6B")
elseif(MODEL_SIZE STREQUAL "1.5b")
    add_compile_definitions(QWEN_SIZE_1_5B)
    set(MODEL_NAME_STR "1.5B")
elseif(MODEL_SIZE STREQUAL "1.7b")
    add_compile_definitions(QWEN_SIZE_1_7B)
    set(MODEL_NAME_STR "1.7B")
elseif(MODEL_SIZE STREQUAL "3b")
    add_compile_definitions(QWEN_SIZE_3B)
    set(MODEL_NAME_STR "3B")
elseif(MODEL_SIZE STREQUAL "4b")
    add_compile_definitions(QWEN_SIZE_4B)
    set(MODEL_NAME_STR "4B")
elseif(MODEL_SIZE STREQUAL "7b")
    add_compile_definitions(QWEN_SIZE_7B)
    set(MODEL_NAME_STR "7B")
elseif(MODEL_SIZE STREQUAL "8b")
    add_compile_definitions(QWEN_SIZE_8B)
    set(MODEL_NAME_STR "8B")
elseif(MODEL_SIZE STREQUAL "14b")
    add_compile_definitions(QWEN_SIZE_14B)
    set(MODEL_NAME_STR "14B")
endif()

# 1b. Compulsory Input: -Dquant=<fp16|int8|int4>. Must match the mode used by
#     export_weights.py, since the benchmark shares ../src and ../include.
if(NOT DEFINED quant)
    message(FATAL_ERROR
        "\n[!] Error: 'quant' variable is compulsory.\n"
        "Please specify the weight precision using -Dquant=<precision>.\n"
        "Available options: fp16, int8, int4\n"
        "This must match the --quantization used by export_weights.py.\n"
        "Example:\n"
        "  cmake -Dmodel=0.5b -Dquant=fp16 -Dgpu_arch=75 ..\n"
    )
endif()

string(TOLOWER "${quant}" QUANT_MODE)

if(QUANT_MODE STREQUAL "int8")
    add_compile_definitions(QUANT_INT8)
elseif(QUANT_MODE STREQUAL "int4")
    add_compile_definitions(QUANT_INT4)
elseif(NOT QUANT_MODE STREQUAL "fp16")
    message(FATAL_ERROR "Invalid quant option '${quant}'. Options are: fp16, int8, int4")
endif()

# 2. Compulsory Input: -Dgpu_arch=<architecture>
if(NOT DEFINED gpu_arch)
    message(FATAL_ERROR
        "\n[!] Error: 'gpu_arch' variable is compulsory.\n"
        "You must specify the target GPU architecture using -Dgpu_arch=<arch>.\n"
        "Examples:\n"
        "  cmake -Dmodel=0.5b -Dquant=fp16 -Dgpu_arch=75 ..      # Turing (e.g., GTX 1660, RTX 2080)\n"
        "  cmake -Dmodel=0.5b -Dquant=fp16 -Dgpu_arch=80 ..      # Ampere (e.g., A100)\n"
        "  cmake -Dmodel=0.5b -Dquant=fp16 -Dgpu_arch=86 ..      # Ampere (e.g., RTX 3080, RTX 3090)\n"
        "  cmake -Dmodel=0.5b -Dquant=fp16 -Dgpu_arch=89 ..      # Ada Lovelace (e.g., RTX 4080, RTX 4090)\n"
        "  cmake -Dmodel=0.5b -Dquant=fp16 -Dgpu_arch=90 ..      # Hopper (e.g., H100)\n"
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