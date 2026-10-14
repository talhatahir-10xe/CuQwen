cmake_minimum_required(VERSION 3.18)
project(cuqwen LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

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

# Map the (family-validated) size string to its compile definition + display name.
# 14b is valid in both families; the ARCH_* define above disambiguates the dims.
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

# 1b. Compulsory Input: -Dquant=<fp16|int8|int4>
#     int8/int4 = weights-only quantization (W8A16/W4A16): linear-projection
#     weights are stored/loaded quantized + FP16 group scales and dequantized
#     in-kernel. int4 additionally packs 2 weights per byte.
#     Must match the --quantization used by export_weights.py.
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
    set(QUANT_NAME_STR "INT8 (weights-only, W8A16)")
elseif(QUANT_MODE STREQUAL "int4")
    add_compile_definitions(QUANT_INT4)
    set(QUANT_NAME_STR "INT4 (weights-only, W4A16)")
elseif(QUANT_MODE STREQUAL "fp16")
    set(QUANT_NAME_STR "FP16")
else()
    message(FATAL_ERROR "Invalid quant option '${quant}'. Options are: fp16, int8, int4")
endif()

# 2. Compulsory Input: -Dgpu_arch=<architecture> (e.g. 75, 80, 86, 89)
if(NOT DEFINED gpu_arch)
    message(FATAL_ERROR 
        "\n[!] Error: 'gpu_arch' variable is compulsory.\n"
        "Please specify your target CUDA architecture using -Dgpu_arch=<arch>.\n"
        "Examples:\n"
        "  Turing (RTX 20xx / GTX 16xx / GTX 1650 Ti / T4):  cmake -Dmodel=0.5b -Dquant=fp16 -Dgpu_arch=75 ..\n"
        "  Ampere (RTX 30xx / A100):                          cmake -Dmodel=0.5b -Dquant=fp16 -Dgpu_arch=86 ..\n"
        "  Ada Lovelace (RTX 40xx / L4):                       cmake -Dmodel=0.5b -Dquant=fp16 -Dgpu_arch=89 ..\n"
    )
endif()

# Set CUDA Architecture for CMake targets
set(CMAKE_CUDA_ARCHITECTURES ${gpu_arch})

message(STATUS "Configuring CUQWEN build for ${ARCH_NAME_STR}-${MODEL_NAME_STR}")
message(STATUS "Weight precision: ${QUANT_NAME_STR}")
message(STATUS "Targeting CUDA Compute Capability: sm_${gpu_arch}")

# 3. Find External Packages (CUDA & PCRE2)
find_package(CUDAToolkit REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(PCRE2 REQUIRED libpcre2-8)

include_directories(
    include
    ${CUDAToolkit_INCLUDE_DIRS}
    ${PCRE2_INCLUDE_DIRS}
)

link_directories(
    ${PCRE2_LIBRARY_DIRS}
)

# Enable fast-math for NVCC
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")

file(GLOB SOURCES 
    "src/*.cpp"
    "src/*.cu"
)

add_executable(cuqwen ${SOURCES})

set_target_properties(cuqwen PROPERTIES
    CUDA_SEPARATE_COMPILATION ON
)

target_link_libraries(cuqwen PRIVATE
    CUDA::cudart
    ${PCRE2_LIBRARIES}
)
