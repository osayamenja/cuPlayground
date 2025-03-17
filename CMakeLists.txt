cmake_minimum_required(VERSION 3.30)

if(DEFINED ENV{NERSC_HOST})
    execute_process(COMMAND "which" "g++"
            COMMAND_ERROR_IS_FATAL ANY
            OUTPUT_VARIABLE CPP_COMP_PATH
            OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND "which" "gcc"
            COMMAND_ERROR_IS_FATAL ANY
            OUTPUT_VARIABLE C_COMP_PATH OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(CMAKE_CXX_COMPILER ${CPP_COMP_PATH})
    set(CMAKE_C_COMPILER ${C_COMP_PATH})
endif ()

project(cuPlayground CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(FINE_CXX_FLAGS "-Wall -Wextra -Wsuggest-attribute=const -fno-strict-aliasing -Wno-sign-compare -v")
set(FINE_CXX_FLAGS "${FINE_CXX_FLAGS} -Wno-unknown-pragmas -Wnull-dereference -Wno-switch -Wfloat-equal")
set(FINE_CXX_FLAGS "${FINE_CXX_FLAGS} -Wduplicated-branches -Wformat=2 -Wno-unused-but-set-parameter")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FINE_CXX_FLAGS}")

set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
set(CMAKE_CUDA_ARCHITECTURES "native")
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xfatbin -compress-all -Xcudafe --display_error_number")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler \"${FINE_CXX_FLAGS}\"")

include(CheckCompilerFlag)
check_compiler_flag(CUDA -t4 NVCC_THREADS)

find_package(CUDAToolkit REQUIRED)

add_executable(cuPlayground main.cu
        processor/tiling.cuh
        auditorium/overlap.cuh
        util.cuh
        mma.cuh
        debug.cuh
        benchmarks.cuh
        auditorium/combine.cuh
        auditorium/scheduling.cuh
        warpScheduler.cuh
        arch.cuh
        torching.cuh
)
set_target_properties(cuPlayground PROPERTIES
	POSITION_INDEPENDENT_CODE ON
        CUDA_SEPARABLE_COMPILATION ON
)

string(SUBSTRING "${CMAKE_CUDA_ARCHITECTURES_NATIVE}" 0 2 COMPUTE_CAPABILITY) # xx-real -> xx
math(EXPR GPU_ARCH "${COMPUTE_CAPABILITY} * 10U" OUTPUT_FORMAT DECIMAL)
message(STATUS "GPU 0 Compute Capability: ${COMPUTE_CAPABILITY}")
#Link torch
#target_precompile_headers(cuPlayground PRIVATE torchInclude.h)
#target_link_libraries(cuPlayground PRIVATE "${TORCH_LIBRARIES}")
#CPM
# set(ENV{CPM_USE_NAMED_CACHE_DIRECTORIES} ON)
set(ENV{CPM_USE_LOCAL_PACKAGES} ON)
set(ENV{CPM_SOURCE_CACHE} "./cmake/cache")
include(cmake/CPM.cmake)

set(CCCL_ENABLE_UNSTABLE ON)
CPMAddPackage(
        NAME CCCL
        GITHUB_REPOSITORY nvidia/cccl
        FORCE 1
        GIT_TAG main # Fetches the latest commit on the main branch
)
if(CCCL_ADDED)
    target_link_libraries(cuPlayground PRIVATE CCCL::CCCL)
endif()

#CUTLASS business
CPMAddPackage(
        NAME CUTLASS
        GITHUB_REPOSITORY nvidia/cutlass
        GIT_TAG main
        DOWNLOAD_ONLY TRUE
        OPTIONS
        "CUTLASS_NVCC_ARCHS=${COMPUTE_CAPABILITY}"
)
if(CUTLASS_ADDED)
    # header-only
    target_include_directories(cuPlayground SYSTEM PRIVATE "${CUTLASS_SOURCE_DIR}/include")
    set(cublasdx_CUTLASS_ROOT "${CUTLASS_SOURCE_DIR}")
endif ()

CPMAddPackage(
        NAME FMT
        GITHUB_REPOSITORY fmtlib/fmt
        GIT_TAG 11.0.2
        DOWNLOAD_ONLY
)
if(FMT_ADDED)
    target_link_libraries(cuPlayground PRIVATE fmt::fmt)
endif ()

CPMAddPackage(
        NAME NVTX3
        GITHUB_REPOSITORY NVIDIA/NVTX
        GIT_TAG v3.1.0-c-cpp
        GIT_SHALLOW TRUE
)

if(NVTX3_ADDED)
    target_link_libraries(cuPlayground PRIVATE nvtx3-cpp)
    string(FIND "$ENV{CMAKE_PREFIX_PATH}" "nvtx3" INDEX)
    if(INDEX EQUAL -1)
        # append nvtx3 to prefix path to remove libtorch error
        set(ENV{CMAKE_PREFIX_PATH} "$ENV{CMAKE_PREFIX_PATH}:${NVTX3_SOURCE_DIR}")
    endif()
endif ()
set(MATHDX_VER 25.01)
CPMFindPackage(
        NAME mathdx
        VERSION "${MATHDX_VER}"
        FIND_PACKAGE_ARGUMENTS "REQUIRED COMPONENTS cublasdx CONFIG"
)

set(USE_SYSTEM_NVTX ON)
set(CAFFE2_USE_CUDNN ON)
find_package(Torch REQUIRED HINTS "$ENV{TORCH_ROOT}")
if(NOT Torch_FOUND)
    message(FATAL_ERROR "LibTorch not found!")
endif ()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
target_link_libraries(cuPlayground PRIVATE "${TORCH_LIBRARIES}")

target_link_libraries(cuPlayground PRIVATE mathdx::cublasdx)
target_link_libraries(cuPlayground PRIVATE CUDA::cudart CUDA::cuda_driver CUDA::nvml CUDA::nvtx3 CUDA::curand_static)
find_package(NVSHMEM REQUIRED HINTS "$ENV{NVSHMEM_HOME}/lib/cmake/nvshmem")
target_link_libraries(cuPlayground PRIVATE nvshmem::nvshmem)

set(LINK_NCCL ON)
if(DEFINED ENV{NERSC_HOST} AND ${LINK_NCCL})
    find_library(NCCL
            NAMES libnccl_static.a
            HINTS "$ENV{NCCL_HOME}"
            REQUIRED
    )
    target_link_libraries(cuPlayground PRIVATE "${NCCL}")
    target_include_directories(cuPlayground SYSTEM PRIVATE "$ENV{NCCL_HOME}/include")
endif ()

# Link Cray's GPU-accelerated MPICH
if(DEFINED ENV{LINK_GTL} AND DEFINED ENV{NERSC_HOST} AND "$ENV{NERSC_HOST}" STREQUAL "perlmutter")
    find_library(GTL
            NAMES libmpi_gtl_cuda.so.0
            HINTS /opt/cray/pe/lib64/
            REQUIRED)
    target_link_libraries(cuPlayground PRIVATE "${GTL}")
endif ()

target_compile_options(cuPlayground PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xfatbin -compress-all>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:-t0; --generate-line-info>
        $<$<COMPILE_LANGUAGE:CUDA>:SHELL:-gencode=arch=compute_${COMPUTE_CAPABILITY},code=sm_${COMPUTE_CAPABILITY}>
)

# Pre-compile constants
# Read JSON file content
file(READ "aristos_config.json" ARISTOS_CONFIG)

string(JSON CAP_FACTOR_V GET ${ARISTOS_CONFIG} "capacity_factor")
message(STATUS "capacity_factor: ${CAP_FACTOR_V}")

string(JSON DROP_TOKENS_V GET ${ARISTOS_CONFIG} "drop_tokens")
message(STATUS "drop_tokens: ${DROP_TOKENS_V}")

string(JSON E_TOP_K_V GET ${ARISTOS_CONFIG} "expert_top_k")
message(STATUS "expert_top_k: ${E_TOP_K_V}")

string(JSON IS_TRAINING_V GET ${ARISTOS_CONFIG} "is_training")
message(STATUS "is_training: ${IS_TRAINING_V}")

string(JSON HIDDEN_ACT_V GET ${ARISTOS_CONFIG} "hidden_act")
message(STATUS "hidden_act: ${HIDDEN_ACT_V}")

string(JSON HIDDEN_SIZE_V GET ${ARISTOS_CONFIG} "hidden_size")
message(STATUS "hidden_size: ${HIDDEN_SIZE_V}")

string(JSON I_SIZE_V GET ${ARISTOS_CONFIG} "intermediate_size")
message(STATUS "intermediate_size: ${I_SIZE_V}")

string(JSON MU_BATCH_V GET ${ARISTOS_CONFIG} "micro_batch")
message(STATUS "micro_batch: ${MU_BATCH_V}")

string(JSON NUM_EXPERTS_V GET ${ARISTOS_CONFIG} "num_experts")
message(STATUS "num_experts: ${NUM_EXPERTS_V}")

string(JSON SEQ_LEN_V GET ${ARISTOS_CONFIG} "sequence_len")
message(STATUS "sequence_len: ${SEQ_LEN_V}")

string(JSON DTYPE_V GET ${ARISTOS_CONFIG} "torch_dtype")
message(STATUS "torch_dtype: ${DTYPE_V}")

target_compile_definitions(cuPlayground
        PRIVATE
        PLAY_ARCH=${GPU_ARCH}
        CAP_FACTOR=${CAP_FACTOR_V}
        DROP_TOKENS=${DROP_TOKENS_V}
        E_TOP_K=${E_TOP_K_V}
        HIDDEN_ACT=${HIDDEN_ACT_V}
        HIDDEN_SIZE=${HIDDEN_SIZE_V}
        I_SIZE=${I_SIZE_V}
        NUM_EXPERTS=${NUM_EXPERTS_V}
        SEQ_LEN=${SEQ_LEN_V}
        DTYPE=${DTYPE_V}
)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(cuPlayground PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-Og;-g;>
            $<$<COMPILE_LANGUAGE:CUDA>:-O1; -g; -G>
    )
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(cuPlayground PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-O3>
            $<$<COMPILE_LANGUAGE:CUDA>:SHELL:-gencode=arch=compute_${COMPUTE_CAPABILITY},code=lto_${COMPUTE_CAPABILITY}>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas -v;--expt-relaxed-constexpr>
    )
endif ()
