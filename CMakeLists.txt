cmake_minimum_required(VERSION 3.5)

project(xllm LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread --std=c++17 -O2 -march=native")

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread --std=c++17 -march=native")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")

option(USE_CUDA "use cuda" OFF)
message(STATUS "USE_CUDA: ${USE_CUDA}")
if (USE_CUDA)
    enable_language(CUDA)
    add_compile_definitions(USE_CUDA)  #  添加预处理宏：USE_CUDA
    # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")
    add_compile_options(-ldl)
    include_directories(include/cuda)
    message(${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
    set(XLLM_CUDA_SOURCES src/cuda/cudadevice.cpp src/cuda/xllm-cuda.cu)
    set(XLLM_LINKED_LIBS ${XLLM_LINKED_LIBS} cublas)
    link_directories(/usr/local/cuda/lib64)
    set(CMAKE_CUDA_ARCHITECTURES "${CUDA_ARCHITECTURES}")
endif()

set(XLLM_CXX_SOURCES src/param.cpp src/data.cpp src/xllm.cpp src/model.cpp
    src/cpu/cpudevice.cpp src/executor.cpp src/device.cpp)

find_package(OpenMP)
include_directories(include)
include_directories(include/cpu)
add_library(xllm OBJECT ${XLLM_CXX_SOURCES} ${XLLM_CUDA_SOURCES})
target_link_libraries(xllm PUBLIC ${XLLM_LINKED_LIBS} OpenMP::OpenMP_CXX)

add_executable(main main.cpp)
target_link_libraries(main xllm)

add_executable(benchmark benchmark/benchmark.cpp)
target_link_libraries(benchmark xllm)

add_executable(benchmark_batch benchmark/benchmark_batch.cpp)
target_link_libraries(benchmark_batch xllm)

add_executable(quant src/quant.cpp)
target_link_libraries(quant xllm)

option(DEVELOPMENT OFF)
if (${DEVELOPMENT})
    message(STATUS "DEVELOPMENT MODE ON")
    add_subdirectory(tests)
endif ()
