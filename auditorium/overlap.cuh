//
// Created by oja7 on 12/18/24.
//

#ifndef OVERLAP_CUH
#define OVERLAP_CUH

#include <cute/tensor.hpp>
#include <cuda/std/type_traits>
#include <cublasdx.hpp>
#include <nvshmemx.h>
#include <nvshmem.h>
#include <host/nvshmemx_api.h> // Makes CLion happy

enum signal : unsigned short {
    NOOP,
    shouldProcess,
    processed,
};

#define STAGES 2U
#define CELLS 2U
template<unsigned int stage=0, typename T>
// Pointer arithmetic on void yields undefined behaviour
requires (stage < STAGES && !cuda::std::is_same_v<T, void>)
CUTE_DEVICE
T* advanceHeap(T* const& __restrict__ buffer, const unsigned int& slotSize,
    const unsigned int& peer) {
    return buffer + slotSize * ((STAGES * peer) + stage);
}

//cublasdx::sm_of<BLAS>::value
template<class GEMM, unsigned short rounds, bool skip=true>
requires (cublasdx::is_complete_blas_execution<GEMM>::value
&& cublasdx::is_supported<GEMM, cublasdx::sm_of<GEMM>::value>::value)
__global__ void overlapKernel(const typename GEMM::a_value_type* __restrict__ inputs,
    const typename GEMM::b_value_type* __restrict__ weights, cuda::std::byte* __restrict__ staging,
    uint64_t* __restrict__ flags, cuda::std::byte* sHeap, __grid_constant__ const unsigned int rank,
    __grid_constant__ const unsigned int world) {
    // The workflow operates as follows,
    // assuming each PE has a weight matrix and starts with an input matrix.
    // 1. At time i A2A to disseminate vector v_i
    // 2. GEMM on all received vectors
    // 3. A2A to reconstitute original vector v_i
    // 3. Process received vector
    // 4. Repeat
    assert(world == gridDim.x);
    assert(gridDim.y * gridDim.z == 1);
    static_assert(signal::processed == STAGES);
    static_assert(cublasdx::size_of<GEMM>::n == cublasdx::size_of<GEMM>::k);

    extern __shared__ __align__(16) char workspace[];
    __shared__ unsigned int bid;
    // Ensures a 32-bit single register is used
    const unsigned int tid = cooperative_groups::thread_block::thread_rank();
    constexpr auto sliceBytes = sizeof(GEMM::c_value_type) * GEMM::c_size;
    if (tid == 0) {
        // grid::block_rank() == peer rank
        bid = cooperative_groups::grid_group::block_rank();
        staging += sliceBytes * bid;
    }
    __threadfence_block();
    __syncthreads();

    // Make global memory tensor
    auto tAgA = cublasdx::make_tensor(inputs, GEMM::get_layout_gmem_a());
    auto tBgB = cublasdx::make_tensor(weights, GEMM::get_layout_gmem_b());
    auto tCgC = cublasdx::make_tensor(inputs, GEMM::get_layout_gmem_c());
    auto [sA, sB, sC] = GEMM::slice_shared_memory(workspace);

    // Make shared memory tensor
    auto tAsA = cublasdx::make_tensor(sA, GEMM::suggest_layout_smem_a());
    auto tBsB = cublasdx::make_tensor(sB, GEMM::suggest_layout_smem_b());
    auto tCsC = cublasdx::make_tensor(sC, GEMM::suggest_layout_smem_c());

    // Load data from global memory tensor to shared memory tensor
    // Note each block has identical copy of weights
    cublasdx::copy<GEMM, cublasdx::suggested_alignment_of<GEMM>::a_alignment>(tAgA, tAsA);
    cublasdx::copy<GEMM, cublasdx::suggested_alignment_of<GEMM>::b_alignment>(tBgB, tBsB);
    cublasdx::copy<GEMM, cublasdx::suggested_alignment_of<GEMM>::c_alignment>(tCgC, tCsC);
    cublasdx::copy_wait();
#if 0
    if (tid == 0) {
        print_tensor(tAsA);
        print_tensor(tBsB);
    }
#endif

    CUTE_UNROLL
    for (unsigned short i = 0; i < rounds; ++i) {
        // upper bound of number of messages per round
        memcpy_async(cooperative_groups::this_thread_block(), staging, BYTE_CAST(sC), sliceBytes);
        wait(cooperative_groups::this_thread_block());
        // Communicate vector to peer
        nvshmemx_putmem_signal_nbi_block(advanceHeap<0>(sHeap, sliceBytes, rank),
            staging, sliceBytes, flags + rank, shouldProcess, NVSHMEM_SIGNAL_SET, bid);
        if (!tid) {
            // Await data arrival
            nvshmem_signal_wait_until(flags + bid, NVSHMEM_CMP_EQ, shouldProcess);
        }
        __syncthreads();

        /// First stage
        // Copy received data to shared memory workspace
        memcpy_async(cooperative_groups::this_thread_block(), BYTE_CAST(sA),
            advanceHeap<0>(sHeap, sliceBytes, bid), sliceBytes);
        wait(cooperative_groups::this_thread_block());
        // Execute GEMM
        GEMM().execute(GEMM::a_value_type(1.0), tAsA, tBsB, GEMM::c_value_type(0.0), tCsC);
        __syncthreads();

#if 0
        if (tid == 0 and bid == 1 and rank == 0) {
            print_tensor(tAsA);
            print_tensor(tBsB);
            print_tensor(tCsC);
        }
#endif

        memcpy_async(cooperative_groups::this_thread_block(), staging, BYTE_CAST(sC), sliceBytes);
        wait(cooperative_groups::this_thread_block());

        // Eagerly communicate computed vector to peer
        nvshmemx_putmem_signal_nbi_block(advanceHeap<1>(sHeap, sliceBytes, rank),
            staging, sliceBytes, flags + world + rank, processed, NVSHMEM_SIGNAL_SET, bid);

        // Second Stage
        if (!tid) {
            // Await data arrival
            nvshmem_signal_wait_until(flags + world + bid, NVSHMEM_CMP_EQ, processed);
        }
        __syncthreads();
        memcpy_async(cooperative_groups::this_thread_block(), BYTE_CAST(sA),
            advanceHeap<1>(sHeap, sliceBytes, bid), sliceBytes);
        wait(cooperative_groups::this_thread_block());

        // Fused GEMM and ReLU
        GEMM().execute(GEMM::a_value_type(1.0), tAsA, tBsB, GEMM::c_value_type(0.0), tCsC,
            cublasdx::identity{}, cublasdx::identity{}, cublasdx::identity{},
            cutlass::epilogue::thread::ReLU<typename GEMM::c_value_type>{});
        __syncthreads();
#if 0
        if (tid == 0 and bid == 1 and rank == 0) {
            print_tensor(tAsA);
            print_tensor(tCsC);
        }
#endif
    }

    // Store final result in global memory, reusing staging
    memcpy_async(cooperative_groups::this_thread_block(), staging, BYTE_CAST(sC), sliceBytes);
}

void overlapPrototype() {
    auto playStream = cudaStreamPerThread;
    // construct GEMM description
    constexpr auto M = 2U;
    constexpr auto N = 2U;
    constexpr auto K = 2U;
    using inputValueType = float;
    using weightValueType = float;
    using outValueType = float;
    // Do y=xA^T
    using GEMM = decltype(cublasdx::Size<M, N, K>()
                          + cublasdx::Precision<inputValueType>()
                          + cublasdx::Type<cublasdx::type::real>()
                          + cublasdx::Arrangement<cublasdx::row_major>()
                          + cublasdx::Function<cublasdx::function::MM>()
                          + cublasdx::SM<800>()
                          + cublasdx::Block());

    // blocks should be equal to n
    nvshmem_init();
    const auto nPes = nvshmem_n_pes();
    const auto rank = nvshmem_my_pe();
    CUTE_CHECK_ERROR(cudaSetDevice(nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE)));
    const auto abSize = (sizeof(GEMM::a_value_type) * GEMM::a_size * nPes)
    + (sizeof(GEMM::b_value_type) * GEMM::b_size);
    cuda::std::byte* ab;
    CUTE_CHECK_ERROR(cudaMallocAsync(&ab, abSize, playStream));
    const auto heapBytes = (sizeof(GEMM::c_value_type) * GEMM::c_size * nPes * (STAGES + 1))
    + (sizeof(uint64_t) * nPes * STAGES);
    static_assert(sizeof(cuda::std::byte) == sizeof(unsigned char));
    auto* sHeap = static_cast<cuda::std::byte*>(nvshmem_calloc(heapBytes, sizeof(cuda::std::byte)));

    auto* data = malloc(abSize);
    int i = 0;
    static_assert(sizeof(inputValueType) == sizeof(weightValueType));
    static_assert(sizeof(inputValueType) == sizeof(outValueType));
    static_assert(sizeof(weightValueType) == sizeof(outValueType));
    auto bookend = GEMM::a_size * nPes;
    for (;i < bookend; ++i) {
        static_cast<inputValueType*>(data)[i] = static_cast<inputValueType>(rank + i);
    }
    bookend = i + GEMM::b_size;
    for (int j = 0; i < bookend; ++i, ++j) {
        static_cast<weightValueType*>(data)[i] = static_cast<weightValueType>(rank + j + 4);
    }

    CUTE_CHECK_ERROR(cudaMemcpyAsync(ab, data, abSize, cudaMemcpyHostToDevice, playStream));
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    CUTE_CHECK_ERROR(cudaEventRecord(start, playStream));
    overlapKernel<GEMM, 1><<<nPes, GEMM::suggested_block_dim,
    GEMM::shared_memory_size, playStream>>>(
        CAST_TO(inputValueType, ab),
        CAST_TO(weightValueType, ab + (sizeof(GEMM::a_value_type) * GEMM::a_size * nPes)),
        sHeap,
        CAST_TO(uint64_t, sHeap + (GEMM::c_size * nPes * sizeof(GEMM::c_value_type))),
        sHeap + (GEMM::c_size * nPes * sizeof(GEMM::c_value_type)) + (sizeof(uint64_t) * nPes * STAGES),
        nvshmem_my_pe(), nPes);
    CUTE_CHECK_ERROR(cudaEventRecord(stop, playStream));
    CUTE_CHECK_ERROR(cudaPeekAtLastError());
    CUTE_CHECK_ERROR(cudaStreamSynchronize(playStream));
    float duration = 0.0f;
    CUTE_CHECK_ERROR(cudaEventElapsedTime(&duration, start, stop));
    fmt::println("Elapsed time {}", duration);
    // Copy matrix C
    CUTE_CHECK_ERROR(cudaMemcpyAsync(data, sHeap, sizeof(GEMM::c_value_type) * GEMM::c_size * nPes,
        cudaMemcpyDeviceToHost, playStream));
    CUTE_CHECK_ERROR(cudaPeekAtLastError());
    CUTE_CHECK_ERROR(cudaStreamSynchronize(playStream));
    nvshmem_free(sHeap);
    nvshmem_finalize();
    CUTE_CHECK_ERROR(cudaEventDestroy(start));
    CUTE_CHECK_ERROR(cudaEventDestroy(stop));
    CUTE_CHECK_ERROR(cudaFreeAsync(ab, playStream));
    CUTE_CHECK_ERROR(cudaPeekAtLastError());
    CUTE_CHECK_ERROR(cudaStreamSynchronize(playStream));
    // print result
    if (rank == 0) {
        print_tensor(make_tensor(static_cast<GEMM::c_value_type*>(data), cute::make_shape(M*nPes, N)));
    }
    free(data);
}

#endif //OVERLAP_CUH
