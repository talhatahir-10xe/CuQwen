cmake_minimum_required(VERSION 3.18)
project(cuqwen LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

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

# 2. Compulsory Input: -Dgpu_arch=<architecture> (e.g. 75, 80, 86, 89)
if(NOT DEFINED gpu_arch)
    message(FATAL_ERROR 
        "\n[!] Error: 'gpu_arch' variable is compulsory.\n"
        "Please specify your target CUDA architecture using -Dgpu_arch=<arch>.\n"
        "Examples:\n"
        "  Turing (RTX 20xx / GTX 16xx / GTX 1650 Ti / T4):  cmake -Dmodel=0.5b -Dgpu_arch=75 ..\n"
        "  Ampere (RTX 30xx / A100):                          cmake -Dmodel=0.5b -Dgpu_arch=86 ..\n"
        "  Ada Lovelace (RTX 40xx / L4):                       cmake -Dmodel=0.5b -Dgpu_arch=89 ..\n"
    )
endif()

# Set CUDA Architecture for CMake targets
set(CMAKE_CUDA_ARCHITECTURES ${gpu_arch})

message(STATUS "Configuring CUQWEN build for model size: Qwen2.5-${MODEL_NAME_STR}")
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
