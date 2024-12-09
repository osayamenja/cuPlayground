#include <iostream>

#include <typeinfo>
#include <cxxabi.h>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>

#include <cooperative_groups/memcpy_async.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda/cmath>
#include <cuda/std/array>
#include <cuda/std/chrono>
#include <cublasdx.hpp>
#include <cuda/std/type_traits>
#include <cute/arch/copy.hpp>
#include <cute/arch/copy_sm80.hpp>
#include <cutlass/epilogue/thread/activation.h>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/gemm/collective/collective_mma.hpp>
#include <fmt/ranges.h>
#include <cuda.h>
#include <cuda/annotated_ptr>
#include <nccl.h>
#include <nvshmemx.h>
#include <nvshmem.h>
#include <host/nvshmemx_api.h> // Makes CLion happy
#include "processor/tiling.cuh"
#include "processor/gemm.cuh"

#include <cuda/semaphore>
#include <cuda/experimental/device.cuh>

#define CAST_TO(T, p) static_cast<T*>(static_cast<void*>(p))
#define BYTE_CAST(p) static_cast<cuda::std::byte*>(static_cast<void*>(p))
#define NANO_TO_MICRO (cuda::std::nano::den / cuda::std::micro::den)

#define NANO_TO_MICRO (cuda::std::nano::den / cuda::std::micro::den)
#if !defined(CHECK_ERROR_EXIT)
#  define CHECK_ERROR_EXIT(e)                                         \
do {                                                           \
    cudaError_t code = (e);                                      \
    if (code != cudaSuccess) {                                   \
    fprintf(stderr, "<%s:%d> %s:\n    %s: %s\n",               \
    __FILE__, __LINE__, #e,                            \
    cudaGetErrorName(code), cudaGetErrorString(code)); \
    fflush(stderr);                                            \
    exit(1);                                                   \
    }                                                            \
} while (0)
#endif

constexpr unsigned int len = 10U;
template<unsigned int n>
__device__ __forceinline__
unsigned int blockManipulation(const cuda::std::array<bool, n>& isRemote,
    const unsigned int& idx) {
    unsigned int numPeers = 0U;
    cuda::std::array<unsigned int, n> peers{};
#pragma unroll
    for(unsigned int i = 0U; i < n; ++i) {
        const bool b = (idx > 0) * !isRemote[i] + isRemote[i] * (idx == 0);
        peers[numPeers] = !b * peers[numPeers] + i * b;
        numPeers += b;
    }
    return numPeers;
}

template<unsigned int n>
__device__ __forceinline__
unsigned int blockManipulationBranch(const cuda::std::array<bool, n>& isRemote,
    const unsigned int& idx) {
    unsigned int numPeers = 0U;
    cuda::std::array<unsigned int, n> peers{};
#pragma unroll
    for(unsigned int i = 0U; i < n; ++i) {
        if ((isRemote[i] && idx == 0) || (!isRemote[i] && idx > 0)) {
            peers[numPeers++] = i;
        }
    }
    return numPeers;
}

template<unsigned int n>
__global__ void benchBranch(const bool* in, __grid_constant__ const unsigned int idx) {
    cuda::std::array<bool, n> isRemote{};
    size_t start, end;
    double duration = 0.0;
#pragma unroll
    for (unsigned int i = 0; i < n; ++i) {
        isRemote[i] = in[i];
    }
    constexpr unsigned int runs = 4;
#pragma unroll
    for (unsigned int i = 0; i < runs; ++i) {
        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(start)::);
        blockManipulation<len>(isRemote, idx);
        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(end)::);
        duration += static_cast<double>(end - start) / static_cast<double>(runs);
    }
    printf("Branch less is %f, res is %u\n", duration, blockManipulation<len>(isRemote, idx));
    duration = 0.0;
#pragma unroll
    for (unsigned int i = 0; i < runs; ++i) {
        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(start)::);
        blockManipulationBranch<len>(isRemote, idx);
        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(end)::);
        duration += static_cast<double>(end - start) / static_cast<double>(runs);
    }
    printf("Branch is %f, res is %u\n", duration, blockManipulationBranch<len>(isRemote, idx));
}

__always_inline
void launchBenchBranch() {
    boost::random::mt19937 rng(cuda::std::chrono::high_resolution_clock::now()
        .time_since_epoch().count());
    const boost::random::uniform_int_distribution<> bits(0,1);
    std::array<bool, len> b{};
    for (unsigned int i = 0; i < len; ++i) {
        b[i] = bits(rng);
    }
    //fmt::println("{}", b);

    bool* bDevice;
    constexpr unsigned int idx = 1U;
    CHECK_ERROR_EXIT(cudaMalloc(&bDevice, sizeof(bool)*len));
    CHECK_ERROR_EXIT(cudaMemcpy(bDevice, b.data(), sizeof(bool)*len, cudaMemcpyHostToDevice));
    benchBranch<len><<<1,1>>>(bDevice, idx);
    CHECK_ERROR_EXIT(cudaPeekAtLastError());
    CHECK_ERROR_EXIT(cudaDeviceSynchronize());
}

struct __align__(16) Args {
    double* sHeap;
    uint64_t* flags;
    double* result;
    unsigned int n;
    unsigned int rank;
    bool remotePresent;
    unsigned int processingRate;

    Args() = default;
    Args(double* _sHeap, uint64_t * _flags,
        double* _result, const unsigned int& _n,
        const unsigned int& _rank, const bool& _remotePresent, const unsigned int& _processingRate)
        : sHeap(_sHeap),
          flags(_flags),
          result(_result),
          n(_n),
          rank(_rank),
          remotePresent(_remotePresent),
          processingRate(_processingRate) {}
};

__constant__ Args b{};
void __global__ testArgs() {
    printf("Args has rank %u, results %f\n", b.rank, b.result[0]);
    b.sHeap[0] = 45.0;
    b.result[0] = 59.0;
    printf("Args has rank %u, results %f\n", b.rank, b.result[0]);
}
#define TO_MB(b) static_cast<double>(b) / (1024.0f*1024.0f)
void testArgsHost() {
    void* p;
    CHECK_ERROR_EXIT(cudaMalloc(&p, sizeof(double)*4));
    CHECK_ERROR_EXIT(cudaMemset(p, 0, sizeof(double)*4));
    const auto a = Args(static_cast<double*>(p),
        static_cast<uint64_t *>(p) + 1,
        static_cast<double*>(p) + 2,
        1, 0, true, 1);
    CHECK_ERROR_EXIT(cudaMemcpyToSymbol(b, &a, sizeof(Args)));
    testArgs<<<1,1>>>();
    CHECK_ERROR_EXIT(cudaPeekAtLastError());
    CHECK_ERROR_EXIT(cudaDeviceSynchronize());
    std::cout << TO_MB(1024*1024) << std::endl;
}

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

void testGEMM() {
    introduction_example<800>();
}

void testConfig() {
    constexpr auto M = 128U;
    constexpr auto N = 128U;
    constexpr auto K = 64U;
    using inputValueType = __half;
    using weightValueType = __half;
    using outValueType = float;
    // Do y=xA^T
    using GEMM = decltype(cublasdx::Size<M, N, K>()
                          + cublasdx::Precision<inputValueType, weightValueType, outValueType>()
                          + cublasdx::Type<cublasdx::type::real>()
                          + cublasdx::Arrangement<cublasdx::row_major>()
                          + cublasdx::Function<cublasdx::function::MM>()
                          + cublasdx::SM<800>()
                          + cublasdx::Block());

    using config = cublasdx::detail::layout_database::optimal_config<THREADS, cublasdx::sm_of<GEMM>::value,
    typename GEMM::a_value_type, cublasdx::arrangement_of<GEMM>::a == cublasdx::arrangement::col_major,
    cublasdx::alignment_of<GEMM>::a,
    typename GEMM::b_value_type, cublasdx::arrangement_of<GEMM>::b == cublasdx::arrangement::col_major,
    cublasdx::alignment_of<GEMM>::b,
    typename GEMM::c_value_type, cublasdx::arrangement_of<GEMM>::c == cublasdx::arrangement::col_major,
    cublasdx::alignment_of<GEMM>::c,
    cublasdx::size_of<GEMM>::m, cublasdx::size_of<GEMM>::n, cublasdx::size_of<GEMM>::k>;
}

template <typename Element, typename ActivationFunction>
requires(cuda::std::is_same_v<Element, cute::half_t> ||
    cuda::std::is_same_v<Element, cute::bfloat16_t> ||
    cuda::std::is_same_v<Element, cute::tfloat32_t> ||
    cuda::std::is_same_v<Element, float> ||
    cuda::std::is_same_v<Element, cute::float_e4m3_t> ||
    cuda::std::is_same_v<Element, cute::float_e5m2_t>)
CUTE_DEVICE
auto fusedAddActivate(Element& accumulator, const Element& term, const ActivationFunction& op) {
    if constexpr (sizeof(Element) >= 4) {
        return op(fma(Element(1.0f), accumulator, term));
    }
    if constexpr(sizeof(Element) == 2) {
        // Half FMA
        if constexpr (cuda::std::is_same_v<Element, cute::half_t>) {
            return op(cute::half_t(__hfma(__half(1.0f), accumulator.to_half(), term.to_half())));
        }
        // bfloat16 FMA
        return op(cute::bfloat16_t(__hfma(__nv_bfloat16(1.0f), accumulator.to_nv_bfloat16(), term.to_nv_bfloat16())));
    }
    return op(accumulator + term);
}

// conversion operators are reinterpret casts, so technically should be free at runtime
// Below is 2.5X faster
template<>
CUTE_DEVICE
auto fusedAddActivate(cute::half_t& accumulator, const cute::half_t& term,
    const cutlass::epilogue::thread::ReLU<cute::half_t>& op) {
    return cute::half_t(__hfma_relu(__half(1.0f),
        accumulator.to_half(), term.to_half()));
}

// Below is 2.5X faster
template<>
CUTE_DEVICE
auto fusedAddActivate(cute::bfloat16_t& accumulator, const cute::bfloat16_t& term,
    const cutlass::epilogue::thread::ReLU<cute::bfloat16_t>& op) {
    return cute::bfloat16_t(__hfma_relu(__nv_bfloat16(1.0f),
        accumulator.to_nv_bfloat16(), term.to_nv_bfloat16()));
}


void __global__ benchFAA() {
    auto val = 0.5_hf;
    __shared__ cute::half_t bias;
    bias = -0.3_hf;
    cutlass::epilogue::thread::ReLU<cute::half_t> op {};
    double devFAATime = 0.0, vanillaFAATime = 0.0;
    CUTE_UNROLL
    for (int i = 0; i < 1024; ++i) {
        uint64_t start = 0, end = 0;
        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(start)::);
        fusedAddActivate(val, bias, op);
        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(end)::);
        assert(val == 0.15_hf);
        devFAATime += static_cast<double>(end - start) / 1024.0;

        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(start)::);
        val = op(val + bias);
        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(end)::);
        assert(val == 0.15_hf);
        vanillaFAATime += static_cast<double>(end - start) / 1024.0;
    }

    printf("Device: %f, Vanilla: %f", devFAATime, vanillaFAATime);
}

__device__ unsigned int syncP = 0;

template<typename ElementA, typename ElementB, typename ElementC, unsigned int Arch>
struct ProcessorGEMM {
    using BlockMM = decltype(cublasdx::Size<BLOCK_M, BLOCK_N, BLOCK_K_FULL>()
                          + cublasdx::Precision<toCDX<ElementA>, toCDX<ElementB>, toCDX<ElementC>>()
                          + cublasdx::Type<cublasdx::type::real>()
                          + cublasdx::Arrangement<cublasdx::row_major, cublasdx::row_major>()
                          + cublasdx::Function<cublasdx::function::MM>()
                          + cublasdx::SM<Arch>()
                          + cublasdx::Block()
                          + cublasdx::BlockDim<THREADS>());
    using blockTiler = cute::Shape<cute::Int<cublasdx::size_of<BlockMM>::m>,
                                    cute::Int<cublasdx::size_of<BlockMM>::n>,
                                    cute::Int<cublasdx::size_of<BlockMM>::k>>;
    using Parameters = CollectiveMMAConfig<BlockMM, LayoutOptimization::UseSwizzle>;
    using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
        typename Parameters::dispatch,
        blockTiler,
        ElementA,
        cute::Underscore,
        ElementB,
        cute::Underscore,
        typename Parameters::mma_t,
        typename Parameters::gCopyA,
        typename Parameters::sLayA,
        typename Parameters::sCopyA,
        cute::identity,
        typename Parameters::gCopyB,
        typename Parameters::sLayB,
        typename Parameters::sCopyB,
        cute::identity
    >;
};

// <=96 registers: <= 5 blocks
template<class BlockMM, typename ActivationOp = cute::identity,
unsigned int sharedSize = 16 * 1024,
typename MatrixA, typename MatrixB, typename MatrixC, typename MatrixD,
typename MatrixAx, typename MatrixBx, typename MatrixCx, typename MatrixDx,
typename ElementA = toCT<typename BlockMM::a_value_type>,
typename ElementB = toCT<typename BlockMM::b_value_type>,
typename ElementC = toCT<typename BlockMM::c_value_type>>
requires (cute::is_tensor_v<MatrixA>
    && cute::is_tensor_v<MatrixB>
    && cute::is_tensor_v<MatrixC>
    && cute::is_tensor_v<MatrixD>
    && cute::is_tensor_v<MatrixAx>
    && cute::is_tensor_v<MatrixBx>
    && cute::is_tensor_v<MatrixCx>
    && cute::is_tensor_v<MatrixDx>
    && cuda::std::is_same_v<typename MatrixC::value_type, typename MatrixAx::value_type>
    && cuda::std::is_same_v<typename MatrixA::value_type, typename MatrixAx::value_type>
    && cuda::std::is_same_v<typename MatrixB::value_type, typename MatrixBx::value_type>
    && cublasdx::is_complete_blas<BlockMM>::value
    && cublasdx::is_supported<BlockMM, cublasdx::sm_of<BlockMM>::value>::value
    && cublasdx::sm_of<BlockMM>::value >= MIN_ARCH
    && sharedSize % THREADS == 0)
__global__ __maxnreg__(128) void deviceCollectiveMMA(
    const MatrixA mA, const MatrixB mB, const MatrixC mC, const MatrixD mD,
    const MatrixAx mAx, const MatrixBx mBx, const MatrixCx mCx, const MatrixDx mDx) {
    static_assert(rank(mA) == 2 && rank(mB) == 2 && rank(mC) == 2 && rank(mD) == 2);
    static_assert(rank(mAx) == 2 && rank(mBx) == 2 && rank(mCx) == 2 && rank(mDx) == 2);
    using Parameters = CollectiveMMAConfig<BlockMM, LayoutOptimization::UseSwizzle>;
    constexpr auto bM = cublasdx::size_of<BlockMM>::m;
    constexpr auto bN = cublasdx::size_of<BlockMM>::n;
    constexpr auto bK = cublasdx::size_of<BlockMM>::k;
    // put in constant memory
    const unsigned int kChunks = cute::ceil_div(cute::get<1>(mC.shape()), bN);

    using blockTiler = cute::Shape<cute::Int<bM>, cute::Int<bN>, cute::Int<bK>>;

    using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
        typename Parameters::dispatch,
        blockTiler,
        ElementA,
        cute::Underscore,
        ElementB,
        cute::Underscore,
        typename Parameters::mma_t,
        typename Parameters::gCopyA,
        typename Parameters::sLayA,
        typename Parameters::sCopyA,
        cute::identity,
        typename Parameters::gCopyB,
        typename Parameters::sLayB,
        typename Parameters::sCopyB,
        cute::identity
    >;

    typename Parameters::mma_t tiledMMA;
    using TilerOut = cute::Shape<cute::Int<bM>, cute::Int<bN>>;
    auto accum = cute::partition_fragment_C(tiledMMA, TilerOut{});
    static_assert(cuda::std::is_same_v<ElementC, typename decltype(accum)::value_type>);
    cute::clear(accum);

    // Get the appropriate blocks for this thread block
    // use problem shape instead, p_MNK = (cute::ceil_div(M, bM), cute::ceil_div(N, bN), K)
    //auto M = cute::get<0>(mC.shape());
    //auto N = cute::get<1>(mC.shape());
    auto cta_coordX = cute::idx2crd(blockIdx.x, cute::Shape(cute::ceil_div(cute::get<0>(mC.shape()), bM),
        cute::ceil_div(cute::get<1>(mC.shape()), bN)));
    auto cta_coord = cute::make_coord(cute::get<0>(cta_coordX), cute::get<1>(cta_coordX), cute::_);
    auto gA = local_tile(mA, blockTiler{}, cta_coord, cute::Step<cute::_1, cute::X,cute::_1>{});  // (BLK_M,BLK_K,k)
    auto gB = local_tile(mB, blockTiler{}, cta_coord, cute::Step< cute::X,cute::_1,cute::_1>{});  // (BLK_N,BLK_K,k)
    auto gC = local_tile(mC, blockTiler{}, cta_coord, cute::Step<cute::_1,cute::_1, cute::X>{});  // (BLK_M,BLK_N)
    auto gD = local_tile(mD, blockTiler{}, cta_coord, cute::Step<cute::_1,cute::_1, cute::X>{});  // (BLK_M,BLK_N)

    auto k_tile_iter = cute::make_coord_iterator(size<2>(gA));
    int k_tile_count = size<2>(gA);

    using ElementD = typename decltype(gD)::value_type;
    extern __shared__ ElementD scratch[];
    typename ProcessorGEMM<ElementA, ElementB, ElementC, 800>::CollectiveMainloop collective_mma;
    collective_mma(
        accum,
        gA,
        gB,
        accum,
        k_tile_iter, k_tile_count,
        cute::Underscore{},
        threadIdx.x,
        CAST_TO(char, scratch));
#if 1
    __syncthreads();

    // Epilogue

    auto tCgC = tiledMMA.get_slice(threadIdx.x).partition_C(gC);
    auto tDgD = tiledMMA.get_slice(threadIdx.x).partition_C(gD);

    // Accounts for GEMMs that accumulate in types differing from input types,
    // given that the result moonlights as the input for the succeeding GEMM.
    auto gCStoreOp = cutlass::NumericConverter<typename decltype(tCgC)::value_type, ElementC>{};
    auto gDLoadOp = cutlass::NumericConverter<ElementC, ElementD>{};

    // Assume unary operator
    ActivationOp epilogueOp{};
    constexpr auto elems = sharedSize / (THREADS * sizeof(ElementD));
    static_assert(size(accum) % elems == 0);
    constexpr auto trips = size(accum) / elems;
    // Leverage compiler packing for half-precision values into one register
    cutlass::AlignedArray<ElementD, elems> rScratch{};

    constexpr auto sCLay = make_layout(cute::Shape<cute::_128, cute::_64>{});
    auto sC = make_tensor(cute::make_smem_ptr(scratch), sCLay);
    auto tCsC = tiledMMA.get_slice(threadIdx.x).partition_C(sC);

    // Prefetch from global to shared memory
    CUTE_UNROLL
    for (int j = 0; j < elems; ++j) {
        scratch[threadIdx.x + j * THREADS] = tDgD(j);
    }

    CUTE_UNROLL
    for (int i = 0; i < trips; ++i) {
        CUTE_UNROLL
        for (int j = 0; j < elems; ++j) {
            rScratch[j] = scratch[threadIdx.x + j * THREADS];
            if (i + 1 < trips) {
                // Eagerly start loads for the next batch
                scratch[threadIdx.x + j * THREADS] = tDgD(j + i * elems);
            }
        }
        // Fused Bias Add and Activation Function on register fragment
        // Also fuses copy to GMEM
        CUTE_UNROLL
        for (int j = 0; j < elems; ++j) {
            tCgC(j + i * elems) = gCStoreOp(fusedAddActivate(accum(j + i * elems),
                gDLoadOp(rScratch[j]), epilogueOp));
        }
    }

    __syncthreads();
    if (!threadIdx.x) {
        __threadfence_system();
    }
    __syncthreads();
#endif

#if 1
    if (cute::thread0()) {
        cute::print_tensor(mD);
        cute::print_tensor(mA);
        cute::print_tensor(mB);
        cute::print_tensor(mC);
    }
#endif
#if 0
    if (cute::thread0) {
        printf("gC(0, 1): %f, gC(1, 0): %f,\nmC(0, 1): %f, mC(1, 0): %f, p[1]: %f\n",
            cute::half_t::convert(gC(0, 1)),
            cute::half_t::convert(gC(1, 0)),
            cute::half_t::convert(mC(0, 1)),
            cute::half_t::convert(mC(1, 0)),
            cute::half_t::convert(gC.data()[1]));
        printf("sC(0): %.2f, sC(0, 1): %f, sC(1, 0): %f,\nsCratch[0]: %f, sCratch[1]: %f, size(sC): %u\n",
            cute::half_t::convert(sC(0, 0)),
            cute::half_t::convert(sC(0, 1)),
            cute::half_t::convert(sC(1, 0)),
            cute::half_t::convert(scratch[0]),
            cute::half_t::convert(scratch[1]),
            cute::size<1>(sC.layout()));
    }
#endif

#if 0
    if (threadIdx.x == 0) {
        // Signal that this tile is available
        atomicAdd(&syncP, 1);
        // Wait until all tiles are ready.
        while (atomicCAS(&syncP, 0U, 0U) % kChunks != 0){}
    }
    __syncthreads();

    // Clear accumulator registers in preparation
    cute::clear(accum);

    gA = local_tile(mAx, blockTiler{}, cta_coord, cute::Step<cute::_1, cute::X,cute::_1>{});  // (BLK_M,BLK_K,k)
    gB = local_tile(mBx, blockTiler{}, cta_coord, cute::Step< cute::X,cute::_1,cute::_1>{});  // (BLK_N,BLK_K,k)
    gC = local_tile(mCx, blockTiler{}, cta_coord, cute::Step<cute::_1,cute::_1, cute::X>{});  // (BLK_M,BLK_N)
    gD = local_tile(mDx, blockTiler{}, cta_coord, cute::Step<cute::_1,cute::_1, cute::X>{});  // (BLK_M,BLK_N)

    auto k_tile_iterX = cute::make_coord_iterator(size<2>(gA));
    k_tile_count = size<2>(gA);

    // Execute next GEMM now
    collective_mma(
        accum,
        gA,
        gB,
        accum,
        k_tile_iterX, k_tile_count,
        cute::Underscore{},
        threadIdx.x,
        CAST_TO(char, scratch));

    // Epilogue
    tDgD = tiledMMA.get_slice(threadIdx.x).partition_C(gD);
    tCgC = tiledMMA.get_slice(threadIdx.x).partition_C(gC);

    CUTE_UNROLL
    for (int i = 0; i < trips; ++i) {
        // Prefetch
        CUTE_UNROLL
        for (int j = 0; j < elems; ++j) {
            rScratch[j] = gDLoadOp(tDgD(j + i * elems));
        }
        // Fused Bias Add on register fragment
        CUTE_UNROLL
        for (int j = 0; j < elems; ++j) {
            tCgC(j + i * elems) = gCStoreOp(accum(j + i * elems) + rScratch[j]);
        }
    }
    __threadfence(); // Ensures writes are visible device-wide
    __syncthreads();
#endif
    // Signal publisher
#if 0
    if (cute::thread0()) {
        cute::print_tensor(mAx);
        cute::print_tensor(mBx);
        cute::print_tensor(mDx);
        cute::print_tensor(mCx);
    }
#endif
}

void testCollective() {
    const auto playStream = cudaStreamPerThread;
    constexpr auto M = 128;
    constexpr auto N = 64;
    constexpr auto K = 64;

    constexpr auto bM = 128;
    constexpr auto bN = 64;
    constexpr auto bK = 8;
    using inputValueType = cute::half_t;
    using outValueType = inputValueType;
    using weightValueType = cute::half_t;
    using accumulateType = float;

    // Do y=xA^T
    using GEMM = decltype(cublasdx::Size<bM, bN, bK>()
                          + cublasdx::Precision<toCDX<inputValueType>, toCDX<weightValueType>, accumulateType>()
                          + cublasdx::Type<cublasdx::type::real>()
                          + cublasdx::Arrangement<cublasdx::row_major, cublasdx::row_major>()
                          + cublasdx::Function<cublasdx::function::MM>()
                          + cublasdx::SM<800>()
                          + cublasdx::Block()
                          + cublasdx::BlockDim<128>());

    constexpr auto aSize = (sizeof(inputValueType) * M * K);
    constexpr auto abSize = aSize + (sizeof(weightValueType) * N * K);
    constexpr auto abcSize = abSize + (sizeof(outValueType) * M * N);
    constexpr auto len = abcSize + (sizeof(inputValueType) * cute::max(N, K));
    cuda::std::byte* abc;
    CUTE_CHECK_ERROR(cudaMallocAsync(&abc, len, playStream));
    CUTE_CHECK_LAST();
    CUTE_CHECK_ERROR(cudaMemsetAsync(abc, 0, len, playStream));
    auto* data = static_cast<cuda::std::byte*>(calloc(len, sizeof(cuda::std::byte)));

    auto mAHost = make_tensor(CAST_TO(inputValueType, data),
        make_layout(cute::make_shape(M, K), cute::make_stride(K, 1)));
    auto mBHost = make_tensor(CAST_TO(weightValueType, data + aSize),
        make_layout(cute::make_shape(N, K), cute::make_stride(K, 1)));

    // Populate bias vector
    CAST_TO(inputValueType, data + abcSize)[0] = static_cast<inputValueType>(1.0);
    CAST_TO(inputValueType, data + abcSize)[1] = static_cast<inputValueType>(2.0);

    mAHost(0, 0) = static_cast<inputValueType>(0.0);
    mAHost(0, 1) = static_cast<inputValueType>(1.0);
    mAHost(1, 0) = static_cast<inputValueType>(2.0);
    mAHost(1, 1) = static_cast<inputValueType>(3.0);

    mBHost(0, 0) = static_cast<weightValueType>(4.0);
    mBHost(0, 1) = static_cast<weightValueType>(5.0);
    mBHost(1, 0) = static_cast<weightValueType>(6.0);
    mBHost(1, 1) = static_cast<weightValueType>(7.0);

    CUTE_CHECK_ERROR(cudaMemcpyAsync(abc, data, len, cudaMemcpyHostToDevice, playStream));

    auto mA = make_tensor(cute::make_gmem_ptr(CAST_TO(inputValueType, abc)),
        make_layout(cute::make_shape(M, K), cute::make_stride(K, 1)));
    auto mB = make_tensor(cute::make_gmem_ptr(
        CAST_TO(weightValueType, abc + aSize)), make_layout(cute::make_shape(N, K), cute::make_stride(K, 1)));
    auto mC = make_tensor(cute::make_gmem_ptr(CAST_TO(inputValueType, abc + abSize)),
        make_layout(cute::make_shape(M, N), cute::make_stride(N, 1)));

    // bias vector (1, N) broadcast to (M, N)
    auto mD = make_tensor(cute::make_gmem_ptr(CAST_TO(inputValueType, abc + abcSize)),
        make_layout(cute::make_shape(M, N), cute::make_stride(0, 1)));

    // Second GEMM
    auto mAx = make_tensor(cute::make_gmem_ptr(CAST_TO(inputValueType, abc + abSize)),
        make_layout(cute::make_shape(M, N), cute::make_stride(N, 1)));
    auto mBx = make_tensor(cute::make_gmem_ptr(
        CAST_TO(weightValueType, abc + aSize)), make_layout(cute::make_shape(K, N), cute::make_stride(N, 1)));
    auto mCx = make_tensor(cute::make_gmem_ptr(CAST_TO(outValueType, abc)),
        make_layout(cute::make_shape(M, K), cute::make_stride(K, 1)));

    auto mDx = make_tensor(cute::make_gmem_ptr(CAST_TO(inputValueType, abc + abcSize)),
        make_layout(cute::make_shape(M, K), cute::make_stride(0, 1)));

    constexpr auto gemmSharedSize = (sizeof(inputValueType) * GEMM::a_size)
        + (sizeof(weightValueType) + GEMM::b_size);
    constexpr auto sharedSize = cute::max(gemmSharedSize * PIPELINE_STAGES, 128 * 32 * 4);
    using activation = cutlass::epilogue::thread::ReLU<accumulateType>;
    deviceCollectiveMMA<GEMM, activation><<<1, 128, sharedSize, playStream>>>
    (mA, mB, mC, mD,
        mAx, mBx, mCx, mDx);
    CUTE_CHECK_LAST();
    CUTE_CHECK_ERROR(cudaFree(abc));
    free(data);
}

template<unsigned int Arch>
__global__ void testArch() {
    printf("%u", 5);
}

void testAlloc() {
    nvshmem_init();
    CUTE_CHECK_ERROR(cudaSetDevice(nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE)));
    auto* p = nvshmem_calloc(4,1);
    auto* pA = nvshmem_malloc(4);
    auto* pAlign = nvshmem_align(16, 4);
    std::cout << ((uintptr_t)p % 16 == 0) << std::endl;
    std::cout << ((uintptr_t)pA % 16 == 0) << std::endl;
    std::cout << ((uintptr_t)pAlign % 16 == 0) << std::endl;
    std::cout << ((uintptr_t)p) << std::endl;
    std::cout << ((uintptr_t)pA) << std::endl;
    std::cout << ((uintptr_t)pAlign) << std::endl;
    nvshmem_free(p);
    nvshmem_free(pA);
    nvshmem_free(pAlign);
    nvshmem_finalize();
}

/*void debugLayout() {
    constexpr auto M = 128;
    constexpr auto N = 128;
    constexpr auto K = 8;
    using inputValueType = cublasdx::tfloat32_t;
    using weightValueType = cublasdx::tfloat32_t;
    using outValueType = float;
    // Has to be (M, K) * (N, K)
    using GEMM = decltype(cublasdx::Size<M, N, K>()
                          + cublasdx::Precision<inputValueType>()
                          + cublasdx::Type<cublasdx::type::real>()
                          + cublasdx::Arrangement<cublasdx::row_major, cublasdx::row_major>()
                          + cublasdx::Function<cublasdx::function::MM>()
                          + cublasdx::SM<800>()
                          + cublasdx::Block());
    constexpr auto bM = cublasdx::size_of<GEMM>::m;
    constexpr auto bN = cublasdx::size_of<GEMM>::n;
    constexpr auto bK = cublasdx::size_of<GEMM>::k;
    using TileShape = cute::Shape<cute::Int<bM>, cute::Int<bN>, cute::Int<bK>>;
    using SmemLayoutAtomB = cute::Layout<cute::Shape<cute::_128, cute::_8>,
    cute::Stride<cute::_1, cute::_128>>;
    using tt = decltype(tile_to_shape(SmemLayoutAtomB{},
      make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}), cute::Int<2>{})));
    using p = CollectiveMMAConfig<GEMM>;
    using swz = SwizzleAtom<cublasdx::arrangement_of<GEMM>::a,
    MiddleSwizzle<GEMM::a_value_type>{}, cublasdx::size_of<GEMM>::k>::swizzleAtom;
    auto sw = composition(cute::Swizzle<3,2,3>{},
                cute::Layout<cute::Shape < cute::_8,cute::_8>,
                       cute::Stride<cute::_8, cute::_1>>{});

    auto swizzle_atom = composition(cute::Swizzle<3,3,3>{},
                                  cute::Layout<cute::Shape <cute::_8,cute::Shape <cute::_8, cute::_8>>,
                                         cute::Stride<cute::_8,cute::Stride<cute::_1,cute::_64>>>{});
    using tt2 = decltype(tile_to_shape(p::sLayA{},
      make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), cute::Int<2>{})));
}*/

__device__ cute::bfloat16_t matABC[3072];
template<class GEMM, typename ElementA, typename ElementB, typename ElementC>
requires (cublasdx::is_complete_blas<GEMM>::value
    && cublasdx::is_supported<GEMM, cublasdx::sm_of<GEMM>::value>::value
    && cublasdx::sm_of<GEMM>::value >= MIN_ARCH)
__global__ void testSharedCopy() {
    __shared__ ElementA sBuf[3072];
    __threadfence_block();
    __syncthreads();
    using Parameters = CollectiveMMAConfig<GEMM>;
    auto copyA = typename Parameters::gCopyA{};
    auto copyB = typename Parameters::gCopyB{};

    auto gmem_thr_copy_A = copyA.get_slice(threadIdx.x);
    auto gmem_thr_copy_B = copyB.get_slice(threadIdx.x);

    constexpr auto M = cublasdx::size_of<GEMM>::m;
    constexpr auto N = cublasdx::size_of<GEMM>::n;
    constexpr auto K = cublasdx::size_of<GEMM>::k;

    auto mA = make_tensor(cute::make_gmem_ptr(matABC), make_layout(cute::make_shape(M, K), cute::make_stride(K, 1)));
    auto mB = make_tensor(cute::make_gmem_ptr(matABC + 1024), make_layout(cute::make_shape(N, K), cute::make_stride(K, 1)));

    auto cta_coord = make_coord(0, 0, cute::_);
    using TileShape = cute::Shape<cute::Int<M>, cute::Int<N>, cute::Int<K>>;
    auto gA = local_tile(mA, TileShape{}, cta_coord, cute::Step<cute::_1, cute::X,cute::_1>{});  // (BLK_M,BLK_K,k)
    auto gB = local_tile(mB, TileShape{}, cta_coord, cute::Step< cute::X,cute::_1,cute::_1>{});  // (BLK_N,BLK_K,k)

    using SmemLayoutA = decltype(tile_to_shape(
      typename Parameters::sLayA{},
      make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), PIPELINE_STAGES)));
    using SmemLayoutB = decltype(tile_to_shape(
        typename Parameters::sLayB{},
        make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}), PIPELINE_STAGES)));

    auto sA = make_tensor(cute::make_smem_ptr(sBuf), SmemLayoutA{}); // (BLK_M,BLK_K,PIPE)
    auto sB = make_tensor(cute::make_smem_ptr(sBuf + 2048), SmemLayoutB{}); // (BLK_N,BLK_K,PIPE)

#if 0
    if (cute::thread0()) {
        print_tensor(sA);
    }
#endif
    auto tAgA = gmem_thr_copy_A.partition_S(gA);                             // (ACPY,ACPY_M,ACPY_K,k)
    auto tAsA = gmem_thr_copy_A.partition_D(sA);                             // (ACPY,ACPY_M,ACPY_K,PIPE)
    auto tBgB = gmem_thr_copy_B.partition_S(gB);                             // (BCPY,BCPY_N,BCPY_K,k)
    auto tBsB = gmem_thr_copy_B.partition_D(sB);                             // (BCPY,BCPY_N,BCPY_K,PIPE)

    auto tiled_mma = typename Parameters::mma_t{};
    using sCa = cute::Copy_Atom<cute::SM75_U32x2_LDSM_N, ElementA>;
    using sCb = cute::Copy_Atom<cute::SM75_U32x2_LDSM_N, ElementB>;

    auto smem_tiled_copy_A = make_tiled_copy_A(sCa{}, tiled_mma);
    auto smem_thr_copy_A   = smem_tiled_copy_A.get_thread_slice(threadIdx.x);

    auto smem_tiled_copy_B = make_tiled_copy_B(sCb{}, tiled_mma);
    auto smem_thr_copy_B   = smem_tiled_copy_B.get_thread_slice(threadIdx.x);

    auto tCsA = smem_thr_copy_A.partition_S(sA);
    auto tCsB = smem_thr_copy_B.partition_S(sB);

    auto tCsA_p = tCsA(cute::_,cute::_,cute::_,0);
    auto tCsB_p = tCsB(cute::_,cute::_,cute::_,0);
    auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);

    auto tCrB = thr_mma.partition_fragment_B(sB(cute::_,cute::_,0));
    auto tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);

    auto tCrA = thr_mma.partition_fragment_A(sA(cute::_,cute::_,0));
    auto tCrA_copy_view  = smem_thr_copy_A.retile_D(tCrA);

#if 0
    if (cute::thread0()) {
        print(tCrA); printf("\n");
        print(size(tCrA)); printf("\n");
        print(tCrA_copy_view); printf("\n");
        print(size(tCrA_copy_view(cute::_, cute::_, 0))); printf("\n");
        print(tCsA_p); printf("\n");
        print(size(tCsA_p(cute::_, cute::_, 0) )); printf("\n");
        printf("-------------------------------------\n");
        print(tCrB); printf("\n");
        print(size(tCrB)); printf("\n");
        print(tCrB_copy_view); printf("\n");
        print(size(tCrB_copy_view(cute::_, cute::_, 0))); printf("\n");
        print(tCsB_p); printf("\n");
        print(size(tCsB_p(cute::_, cute::_, 0) )); printf("\n");
    }
    if (cute::thread0()) {
        cute::print_tensor(tCsA_p(cute::_,cute::_,0));
        cute::print_tensor(tCrA);
        printf("-------------------------------------\n");
    }
#endif
    cute::copy(smem_tiled_copy_A, tCsA_p(cute::_,cute::_,0), tCrA_copy_view(cute::_,cute::_,0));
    cute::copy(smem_tiled_copy_B, tCsB_p(cute::_,cute::_,0), tCrB_copy_view(cute::_,cute::_,0));

    if (cute::thread0()) {
        cute::print_tensor(tCsA_p(cute::_,cute::_,0));
        cute::print_tensor(tCrA);
    }
}

/*template<class GEMM>
requires (cublasdx::is_complete_blas<GEMM>::value
    && cublasdx::is_supported<GEMM, cublasdx::sm_of<GEMM>::value>::value
    && cublasdx::sm_of<GEMM>::value >= MIN_ARCH)
__global__ void testGCopy() {
    __shared__ cute::tfloat32_t sBuf[3072];
    if (cute::thread0()) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            sBuf[i] = cute::tfloat32_t(0.01);
            sBuf[2048 + i] = cute::tfloat32_t(0.03);
        }
    }
    __threadfence_block();
    __syncthreads();
    using Parameters = CollectiveMMAConfig<GEMM>;
    auto copyA = typename Parameters::gCopyA{};
    auto copyB = typename Parameters::gCopyB{};

    auto gmem_thr_copy_A = copyA.get_slice(threadIdx.x);
    auto gmem_thr_copy_B = copyB.get_slice(threadIdx.x);

    constexpr auto M = cublasdx::size_of<GEMM>::m;
    constexpr auto N = cublasdx::size_of<GEMM>::n;
    constexpr auto K = cublasdx::size_of<GEMM>::k;

    auto mA = make_tensor(cute::make_gmem_ptr(matABC), make_layout(cute::make_shape(M, K), cute::make_stride(K, 1)));
    auto mB = make_tensor(cute::make_gmem_ptr(matABC + 1024), make_layout(cute::make_shape(N, K), cute::make_stride(K, 1)));
    auto mC = make_tensor(cute::make_gmem_ptr(matABC + 1536), make_layout(cute::make_shape(M, N), cute::make_stride(1, M)));

    auto cta_coord = make_coord(0, 0, cute::_);
    using TileShape = cute::Shape<cute::Int<M>, cute::Int<N>, cute::Int<K>>;
    auto gA = local_tile(mA, TileShape{}, cta_coord, cute::Step<cute::_1, cute::X,cute::_1>{});
    auto gB = local_tile(mB, TileShape{}, cta_coord, cute::Step< cute::X,cute::_1,cute::_1>{});
    auto gC = local_tile(mC, TileShape{}, cta_coord, cute::Step<cute::_1,cute::_1, cute::X>{});

    using SmemLayoutA = decltype(tile_to_shape(
      typename Parameters::sLayA{},
      make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), PIPELINE_STAGES)));
    using SmemLayoutB = decltype(tile_to_shape(
        typename Parameters::sLayB{},
        make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}), PIPELINE_STAGES)));

    auto sA = make_tensor(cute::make_smem_ptr(sBuf), SmemLayoutA{});
    auto sB = make_tensor(cute::make_smem_ptr(sBuf + 2048), SmemLayoutB{});

    if (cute::thread0()) {
        print_tensor(sA);
    }

    auto tAgA = gmem_thr_copy_A.partition_S(gA);
    auto tAsA = gmem_thr_copy_A.partition_D(sA);
    auto tBgB = gmem_thr_copy_B.partition_S(gB);
    auto tBsB = gmem_thr_copy_B.partition_D(sB);

    copy(copyB, tBgB(cute::_,cute::_,cute::_,0), tBsB(cute::_,cute::_,cute::_,0));
}*/

template<typename T>
void printType() {
    // Get the mangled name
    const char* mangledName = typeid(T).name();

    // Demangle the name
    int status;
    char* demangledName = abi::__cxa_demangle(mangledName, nullptr, nullptr, &status);

    // Print the demangled name
    if (status == 0) {
        std::cout << "Demangled name: " << demangledName << std::endl;
    } else {
        std::cerr << "Demangling failed!" << std::endl;
    }
    // Free the memory allocated by abi::__cxa_demangle
    free(demangledName);
}

void golfing() {
    constexpr auto M = 128;
    constexpr auto N = 64;
    constexpr auto K = 8;
    using inputValueType = cute::bfloat16_t;
    using weightValueType = cute::bfloat16_t;
    using outValueType = float;
    // Has to be (M, K) * (N, K)
    using GEMM = decltype(cublasdx::Size<M, N, K>()
                          + cublasdx::Precision<toCDX<inputValueType>, toCDX<weightValueType>, outValueType>()
                          + cublasdx::Type<cublasdx::type::real>()
                          + cublasdx::Arrangement<cublasdx::row_major, cublasdx::row_major>()
                          + cublasdx::Function<cublasdx::function::MM>()
                          + cublasdx::SM<800>()
                          + cublasdx::Block());
    testSharedCopy<GEMM, inputValueType, weightValueType, outValueType><<<1,1>>>();
    CUTE_CHECK_LAST();
}

void testBiasTrick() {
    cute::array<float, 4> a{{0, 1, 2, 3}};
    auto t = make_tensor(cute::make_gmem_ptr(a.data()), make_layout(cute::make_shape(2,2), cute::LayoutRight{}));
    print_tensor(t);
    cute::array<float, 2> b{{4, 5}};
    auto bias = make_tensor(b.data(), make_layout(cute::make_shape(2,2), cute::make_stride(0, 1)));
    axpby(1.0f, bias, 1.0f, t);
    print_tensor(bias);
    print_tensor(t);
}

template<typename UnaryActivationOp>
void testActivation() {
    UnaryActivationOp op{};
    constexpr auto g = -0.1f;
    printf("Pre Val: %f, Post Val: %f", g, op(g));
}

template<unsigned int threads = 128, unsigned int buf = 4 * 1024>
requires(buf > threads && buf % threads == 0)
__global__ void benchBankConflict() {
    assert(blockDim.x * blockDim.y * blockDim.z == threads);
    size_t start = 0, end = 0;
    double freeTime = 0.0, blockedTime = 0.0;
    __shared__ unsigned int db[buf];
    constexpr auto elems = buf / threads;
    const unsigned int tid = cooperative_groups::thread_block::thread_rank();
    unsigned int x = 0;
    #pragma unroll
    for (int i = 0; i < 1024; ++i) {
        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(start)::);
        // Minimizes bank-free conflict
        #pragma unroll
        for (int j = 0; j < elems; ++j) {
            // Write
            db[tid + j * threads] = tid;
            // Read
            x += db[tid + j * threads];
        }
        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(end)::);
        freeTime += static_cast<double>(end - start) / 1024.0;

        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(start)::);
        // Blocked access
        #pragma unroll
        for (int j = 0; j < elems; ++j) {
            // Write
            db[j + tid * elems] = tid;
            // Read
            x += db[j + tid * elems];
        }
        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(end)::);
        blockedTime += static_cast<double>(end - start) / 1024.0;
    }
    if (tid == 0) {
        printf("Free: %f, Block: %f\n", freeTime, blockedTime);
    }
}

#define N_FOO 8
__device__ unsigned long int foo[N_FOO];
__global__ void testAtomicMax() {
    const unsigned int tid = cooperative_groups::thread_block::thread_rank();
    const unsigned int bid = cooperative_groups::grid_group::block_rank();
    const unsigned int nB = cooperative_groups::grid_group::num_blocks();
    #pragma unroll
    for (unsigned int i = tid; i < N_FOO; i += THREADS) {
        cuda::std::ignore = cuda::atomic_ref<unsigned long int, cuda::thread_scope_device>{foo[i]}.fetch_max(bid);
    }
    __syncthreads();
    if (bid == 0) {
        #pragma unroll
        for (unsigned int i = tid; i < N_FOO; i += THREADS) {
            if (foo[i] != nB - 1) {
                printf("foo[%u]: %lu is wrong\n", i, foo[i]);
                assert(false);
            }
        }
    }
}
void hostAMax() {
    auto* p = calloc(N_FOO, sizeof(unsigned long int));
    // Sets foo to zero
    CUTE_CHECK_ERROR(cudaMemcpyToSymbol(foo, p, N_FOO * sizeof(unsigned long int)));
    testAtomicMax<<<64, THREADS>>>();
    CUTE_CHECK_LAST();
    free(p);
}

__device__ unsigned int counter[8] = {400U, 400U, 400U, 400U, 400U, 400U, 400U, 400U};
__device__ unsigned int baton = 0U;
__device__ cuda::std::atomic_flag interrupt{false};

__global__ __maxnreg__(128) void atoEx(unsigned int* p) {
    if (blockIdx.x == 0) {
        // producer
        for (int i = 0; i < 8; ++i) {
            for (unsigned int j = threadIdx.x; j < 401; j += 128) {
                atomicExch(p + j, i + 1);
            }
        }
    }
    else {
        if (threadIdx.x == 0) {
            // consumer
            float durationEx = 0.0;
            for (int i = 0; i < 8; ++i) {
                size_t start, end;
                unsigned int x = 0U, next=0U;
                asm volatile("mov.u64 %0, %%globaltimer;": "=l"(start)::);
                // pass on
                while (atomicOr(p + blockIdx.x, 0U) == i) {}
                asm volatile("mov.u64 %0, %%globaltimer;": "=l"(end)::);
                durationEx += static_cast<float>(end - start) / 8.0f;
            }
            printf("Block %u, V: %u, AtoEx Val: %f\n", blockIdx.x, atomicOr(p + blockIdx.x, 0U), durationEx);
        }
    }
}

void atoExHost() {
    void* p;
    constexpr std::array<unsigned int, 400> arr{};
    CUTE_CHECK_ERROR(cudaMallocAsync(&p, sizeof(unsigned int)*arr.size(), cudaStreamPerThread));
    CUTE_CHECK_ERROR(cudaMemsetAsync(p, 0, sizeof(unsigned int)*arr.size(),
        cudaStreamPerThread));
    atoEx<<<401,128,0,cudaStreamPerThread>>>(static_cast<unsigned int*>(p));
    CUTE_CHECK_LAST();
}

void __global__ benchShared(float in) {
    extern __shared__ cuda::std::byte pad[];
    bool* interrupt = CAST_TO(bool, pad);
    if (cooperative_groups::thread_block::thread_rank() == 0) {
        *interrupt = false;
    }
    for (unsigned int i = threadIdx.x; i < 4096; i += 128) {
        CAST_TO(float, pad)[i] = 0.0f;
    }
    __syncthreads();

    while (!*interrupt) {
        for (unsigned int i = threadIdx.x; i < 4096; i += 128) {
            CAST_TO(float, pad)[i] += 0.1f;
        }
        if (cooperative_groups::thread_block::thread_rank() == 0) {
            interrupt[0] = CAST_TO(float, pad)[0] > in;
        }
        __syncthreads();
    }
}

void persisting() {
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "L2 Cache: " << prop.l2CacheSize << " L2 Persist Window: " << prop.accessPolicyMaxWindowSize << std::endl;
    int dev = 0, devAttr = 0;
    CUTE_CHECK_ERROR(cudaDeviceGetAttribute(&devAttr, cudaDevAttrL2CacheSize, dev));
}

__global__ void benchPersist(unsigned int* p) {
    constexpr auto rounds = 1024U;
    const auto tid = cooperative_groups::thread_block::thread_rank();
    float duration = 0.0;
    for (int i = 0; i < rounds; ++i) {
        size_t start, end;
        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(start)::);
        for (unsigned int j = 0; j < 128 * 128; j += 128) {
            auto x = p[j];
            p[j] = tid + static_cast<unsigned int>(x * sinf(static_cast<float>(tid)));
        }
        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(end)::);
        duration += static_cast<float>(end - start) / (static_cast<float>(NANO_TO_MICRO) * rounds);
    }
    __syncthreads();
    auto result = CAST_TO(float, p);
    result[0] = 0.0;
    cuda::std::ignore = cuda::atomic_ref{result[0]}.fetch_max(duration);
    __syncthreads();
    if (tid == 0) {
        printf("Time to do work is %f\n", result[0]);
        result[0] = 0.0;
    }
}

void streamPersist(void* p, const unsigned long& bytes) {
    cudaStreamAttrValue stream_attribute;   // Stream level attributes data structure
    stream_attribute.accessPolicyWindow.base_ptr  = p; // Global Memory data pointer
    // Number of bytes for persistence access.
    stream_attribute.accessPolicyWindow.num_bytes = bytes;
    // (Must be less than cudaDeviceProp::accessPolicyMaxWindowSize)
    stream_attribute.accessPolicyWindow.hitRatio  = 1.0;                          // Hint for cache hit ratio
    stream_attribute.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting; // Type of access property on cache hit
    stream_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;  // Type of access property on cache miss.

    //Set the attributes to a CUDA stream of type cudaStream_t
    cudaStreamSetAttribute(cudaStreamPerThread, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);
}

__constant__ __inline__ unsigned int x = 80U;
template<typename T>
__global__ __maxnreg__(32) void benchContention(T* p, const __grid_constant__ T top) {
    if (cooperative_groups::thread_block::thread_rank() == 0) {
        constexpr auto rounds = 1024U;
        float duration = 0.0;
        __shared__ float scratch;
        scratch = 0.0f;
        for (int i = 0; i < rounds; ++i) {
            size_t start, end;
            asm volatile("mov.u64 %0, %%globaltimer;": "=l"(start)::);
            atomicExch(p, 23UL);
            asm volatile("mov.u64 %0, %%globaltimer;": "=l"(end)::);
            duration += static_cast<float>(end - start) / rounds;
        }
        cuda::std::ignore = cuda::atomic_ref{scratch}.fetch_max(duration);
        printf("Block %lu: Time to do work is %f and x is %u\n",
            cooperative_groups::grid_group::block_rank(), scratch, x);
    }
}

void benchContentionHost() {
    using at = unsigned long long int;
    constexpr at t = (431 * 1024) + 1;
    void* p;
    CUTE_CHECK_ERROR(cudaMalloc(&p, sizeof(at)));
    CUTE_CHECK_ERROR(cudaMemset(p, 0, sizeof(at)));
    benchContention<<<431, 128>>>(static_cast<at*>(p), t);
    CUTE_CHECK_ERROR(cudaMemset(p, 0, sizeof(at)));
    CUTE_CHECK_ERROR(cudaDeviceSynchronize());
    printf("--------------------------------------------------\n");
    streamPersist(p, sizeof(at));
    benchContention<<<431, 128, 0, cudaStreamPerThread>>>(static_cast<at *>(p), t);
    CUTE_CHECK_LAST();
    at x;
    CUTE_CHECK_ERROR(cudaMemcpy(&x, p, sizeof(at), cudaMemcpyDeviceToHost));
    printf("p is %lu", x);
}

__device__ unsigned int res = 0U;
__global__ void slowMod(unsigned int rounds) {
    auto z = 0U;
    float duration = 0.0f;
    for (int i = 0; i < rounds; ++i) {
        size_t start, end;
        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(start)::);
        z = i % gridDim.x;
        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(end)::);
        duration += static_cast<float>(end - start) / rounds;
    }
    atomicExch(&res, z);

    if (cooperative_groups::grid_group::block_rank() == 0) {
        printf("slowMod takes: %f\n", duration);
    }
}

template<unsigned int p>
__global__ void fastMod(unsigned int rounds) {
    auto x = 0U;
    float duration = 0.0f;
    for (unsigned int i = 0; i < rounds; ++i) {
        size_t start, end;
        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(start)::);
        x = i % p;
        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(end)::);
        duration += static_cast<float>(end - start) / rounds;
    }
    atomicExch(&res, x);
    if (cooperative_groups::grid_group::block_rank() == 0) {
        printf("FastMod takes: %f\n", duration);
    }

}

void testModHost() {
    volatile int x = 800;
    volatile uint rounds = 1024;
    slowMod<<<x, 1>>>(rounds);
    CUTE_CHECK_LAST();
}

__device__ unsigned int q[3];
constexpr unsigned int bb = 4096;
__device__ unsigned int qq[bb];
__device__ unsigned int pDB[bb];
template<unsigned int p>
__global__ void testScheduler(unsigned int bound, bool skip = true) {
    if (threadIdx.x == 0) {
        __shared__ unsigned int up;
        __shared__ unsigned int rQ[p];
        #pragma unroll
        for (int i = 0; i < p; ++i) {
            rQ[i] = i;
        }
        up = bound;
        unsigned int sX = 0U;
        unsigned int tail = 0U;
        size_t start = 0, end = 0;
        q[0] = bound;
        q[1] = bound;
        q[2] = 0U;
        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(start)::);
        while (sX < atomicOr_block(&up, 0U)) {
            auto x = atomicOr(q, 0U) - sX;
            while (x > 0) {
                auto y = atomicOr(q + 1, 0U) - tail;
                while ( y > 0 && x > 0) {
                    auto r = qq[tail];
                    ++sX;
                    --x;
                    ++tail;
                    --y;
                    atomicExch(pDB + r, sX);
                }
            }
            // while (atomicOr(q, 0U) > sX && atomicOr(q + 1, 0U) > tail) {
            //     ++sX;
            //     ++tail;
            // }
        }
        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(end)::);
        if (!skip) {
            printf("single takes: %f\n", static_cast<float>(end - start) / 1e3);
        }

    }
}

__global__ void benchAtAdd(const unsigned int __grid_constant__ bound, bool skip = true) {
    if (!threadIdx.x) {
        q[0] = 0U;
    }
    __syncthreads();
    float d = 0.0f, d2 = 0.0f;
    for (int i = 0; i < bound; ++i) {
        size_t start = 0, end = 0;
        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(start)::);
        atomicAdd(q, 1U);
        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(end)::);
        d += static_cast<float>(end - start) / static_cast<float>(bound);
        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(start)::);
        atomicInc(q, UINT32_MAX);
        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(end)::);
        d2 += static_cast<float>(end - start) / static_cast<float>(bound);
    }
    __syncthreads();
    if (!threadIdx.x && !skip) {
        printf("atAdd takes: %f, atInc takes: %f, val is %u, *val++ is %u\n", d, d2, *q, ++*q);
        *q += 1;
        printf("val: %u", *q);
    }
}

void hostBenchAtAdd() {
    volatile unsigned int b = 4096;
    assert(b == bb);
    for (int i = 0; i < 128; ++i) {
        benchAtAdd<<<1,864>>>(b);
    }
    benchAtAdd<<<1,864>>>(b, false);
    CUTE_CHECK_LAST();
}

void hostSc() {
    CUTE_CHECK_ERROR(cudaSetDevice(1));
    volatile unsigned int b = 4096;
    for (int i = 0; i < 128; ++i) {
        testScheduler<bb><<<1,1>>>(b);
    }
    testScheduler<bb><<<1,1>>>(b, false);
    CUTE_CHECK_LAST();
}

void mmaGolfing() {
    constexpr auto mma = cute::TiledMMA<
          cute::MMA_Atom<cute::SM80_16x8x8_F32F16F16F32_TN>,
          cute::Layout<cute::Shape<cute::_2, cute::_2, cute::_1>>,
        cute::Tile<cute::_64, cute::_32, cute::_8>
        >{};
    print_latex(mma);
}

__global__ void expBench() {
    constexpr auto k = 64;
    cutlass::AlignedArray<float, cute::max(k, 32)> rScratch{};
    __shared__ float gateScratch[128];
    if (!threadIdx.x) {
        #pragma unroll
        for (uint i = 0; i < 128; ++i) {
            gateScratch[i] = static_cast<float>(i);
        }
    }
    __syncthreads();

    /*using sCLay = cute::Layout<cute::Shape<cute::Int<64>, cute::Int<2>>>;*/
    using sCLayX = cute::Layout<cute::Shape<cute::Int<64>, cute::Int<2>>, cute::Stride<cute::_2, cute::_1>>;
    constexpr auto l = make_layout(cute::Shape<cute::Int<64>, cute::Int<2>>{}, cute::LayoutRight{});
    /*using sCLay2 = cute::Layout<cute::Shape<cute::Int<2>, cute::Int<64>>>;
    using sCLay2X = cute::Layout<cute::Shape<cute::Int<2>, cute::Int<64>>, cute::Stride<cute::_64, cute::_1>>;*/
    /*print_tensor(make_tensor(cute::make_smem_ptr(gateScratch), sCLay{}));
    printf("t(1): %f\n", make_tensor(cute::make_smem_ptr(gateScratch), sCLay{})(1));*/
    auto t = make_tensor(cute::make_smem_ptr(gateScratch), l);
    /*print_tensor(t);
    printf("t(1): %f\n", make_tensor(cute::make_smem_ptr(gateScratch), sCLayX{})(1));*/
    /*print_tensor(make_tensor(cute::make_smem_ptr(gateScratch), sCLay2{}));
    printf("t(1): %f\n", make_tensor(cute::make_smem_ptr(gateScratch), sCLay2{})(1));
    print_tensor(make_tensor(cute::make_smem_ptr(gateScratch), sCLay2X{}));
    printf("t(1): %f\n", make_tensor(cute::make_smem_ptr(gateScratch), sCLay2X{})(1));*/
}

__device__ __forceinline__
bool isLikelyRegister(void* p) {
    return !(__isShared(p) &&
        __isLocal(p) &&
        __isConstant(p) &&
        __isGlobal(p) &&
        __isGridConstant(p));
}

template<typename BlockMM>
__global__ __maxnreg__(128) void regPair() {
    cutlass::AlignedArray<cuda::std::pair<float, unsigned int>, 32> f{};
    #pragma unroll
    for (int i = 0; i < f.size(); ++i) {
        const bool isRegister = !(__isShared(f.data() + i) &&
        __isLocal(f.data() + i) &&
        __isConstant(f.data() + i) &&
        __isGlobal(f.data() + i) &&
        __isGridConstant(f.data() + i));
        assert(isRegister);
    }
    assert(isLikelyRegister(&blockIdx.x));
    constexpr auto t = BlockMM::block_dim.x;
    static_assert(BlockMM::block_dim.x == 128);
}

struct FooBarrier {
    cuda::barrier<cuda::thread_scope_device>* deviceBarrier;
    CUTE_HOST_DEVICE
    FooBarrier() = default;

    CUTE_HOST
    explicit FooBarrier(cuda::barrier<cuda::thread_scope_device>* _deviceBarrier):
    deviceBarrier(_deviceBarrier){}
};

__constant__ FooBarrier fDevice;
template<unsigned int blocks>
__global__ void gridBarrier(cuda::barrier<cuda::thread_scope_device>* deviceBarrier, bool skip = true) {
    float d = 0.0f;
    constexpr auto rounds = 64;
    for (unsigned int i = 0; i < rounds; ++i) {
        size_t start = 0, end = 0;
        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(start)::);
        if (!threadIdx.x) {
            if ((atomicAdd(q, 1U) + 1) % blocks == 0) {
                atomicAdd(q + 1, 1U);
            }
            while (atomicOr(q + 1, 0U) != i + 1){}
        }
        __syncthreads();
        /*if (!threadIdx.x) {
            fDevice.deviceBarrier->arrive_and_wait();
        }
        __syncthreads();*/
        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(end)::);
        d += static_cast<float>(end - start) / static_cast<float>(rounds);
    }
    if (!skip && !threadIdx.x) {
        printf("Block %u says Blockade takes: %f and q is %u\n", blockIdx.x, d, *q);
    }
}

void hostB() {
    CUTE_CHECK_ERROR(cudaSetDevice(1));
    constexpr std::array<unsigned int, 2> qHost{{0U, 0U}};
    CUTE_CHECK_ERROR(cudaMemcpyToSymbol(q, qHost.data(), qHost.size()*sizeof(unsigned int)));
    const auto host_b = new cuda::barrier<cuda::thread_scope_device>{256};
    cuda::barrier<cuda::thread_scope_device>* b;
    CUTE_CHECK_ERROR(cudaMalloc(&b, sizeof(cuda::barrier<cuda::thread_scope_device>)));
    CUTE_CHECK_LAST();
    CUTE_CHECK_ERROR(cudaMemcpy(b, host_b, sizeof(cuda::barrier<cuda::thread_scope_device>), cudaMemcpyHostToDevice));
    CUTE_CHECK_LAST();

    const auto f = FooBarrier(b);
    CUTE_CHECK_ERROR(cudaMemcpyToSymbol(fDevice, &f, sizeof(cuda::barrier<cuda::thread_scope_device>)));
    gridBarrier<256><<<256, 128>>>(b, false);
    CUTE_CHECK_LAST();
    delete host_b;
}
int main() {
    testCollective();
    CUTE_CHECK_LAST();
}