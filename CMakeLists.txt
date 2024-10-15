cmake_minimum_required(VERSION 3.30)
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

find_package(Torch REQUIRED HINTS "$ENV{TORCH_ROOT}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(cuPlayground main.cu
        processor/gemm.cuh)
set_target_properties(cuPlayground PROPERTIES
	POSITION_INDEPENDENT_CODE ON
        CUDA_SEPARABLE_COMPILATION ON
)

#Link torch
target_link_libraries(cuPlayground PRIVATE "${TORCH_LIBRARIES}")
#CPM
set(ENV{CPM_USE_LOCAL_PACKAGES} ON)
set(ENV{CPM_SOURCE_CACHE} "./cmake/cache")
include(cmake/CPM.cmake)

set(CCCL_ENABLE_UNSTABLE ON)
CPMAddPackage(
        NAME CCCL
        GITHUB_REPOSITORY nvidia/cccl
        GIT_TAG main # Fetches the latest commit on the main branch
)
if(CCCL_ADDED)
    target_link_libraries(cuPlayground PRIVATE CCCL::CCCL)
endif()

CPMAddPackage(
        NAME FMT
        GITHUB_REPOSITORY fmtlib/fmt
        GIT_TAG 11.0.2
        DOWNLOAD_ONLY
)
if(FMT_ADDED)
    target_include_directories(cuPlayground SYSTEM PRIVATE "${FMT_SOURCE_DIR}/include")
endif ()

CPMAddPackage(
        NAME SPDLOG
        GITHUB_REPOSITORY gabime/spdlog
        VERSION 1.14.1
        OPTIONS "SPDLOG_FMT_EXTERNAL 1"
)
if(SPDLOG_ADDED)
    target_link_libraries(cuPlayground PRIVATE spdlog::spdlog)
endif ()

set(MATHDX_VER 24.08)
set(MATHDX_URL_PREFIX "https://developer.download.nvidia.com/compute/cublasdx/redist/cublasdx")
set(MATHDX_URL "${MATHDX_URL_PREFIX}/nvidia-mathdx-${MATHDX_VER}.0.tar.gz")
CPMFindPackage(
        NAME mathdx
        VERSION "${MATHDX_VER}"
        URL "${MATHDX_URL}"
        FIND_PACKAGE_ARGUMENTS "REQUIRED COMPONENTS cublasdx CONFIG"
)

target_link_libraries(cuPlayground PRIVATE mathdx::cublasdx)
target_link_libraries(cuPlayground PRIVATE CUDA::cudart CUDA::cuda_driver CUDA::nvml CUDA::nvtx3)
find_package(NVSHMEM REQUIRED HINTS "$ENV{NVSHMEM_HOME}/lib/cmake/nvshmem")
target_link_libraries(cuPlayground PRIVATE nvshmem::nvshmem)

target_compile_options(cuPlayground PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xfatbin -compress-all>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas -v;--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:-t0; --generate-line-info>
        # Required to support std::tuple in device code
)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(cuPlayground PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-Og;-g;>
            $<$<COMPILE_LANGUAGE:CUDA>:-O0; -g; -G>
    )
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(cuPlayground PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-O3>
            $<$<COMPILE_LANGUAGE:CUDA>:-O3>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas -O3>
    )
endif ()
