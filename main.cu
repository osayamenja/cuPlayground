#include <iostream>

#include <boost/random/uniform_int_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>

#include <cooperative_groups/memcpy_async.h>
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
#include <nccl.h>
#include <nvshmemx.h>
#include <nvshmem.h>
#include <host/nvshmemx_api.h> // Makes CLion happy
#include "processor/tiling.cuh"
#include "processor/gemm.cuh"

#define CAST_TO(T, p) static_cast<T*>(static_cast<void*>(p))
#define BYTE_CAST(p) static_cast<cuda::std::byte*>(static_cast<void*>(p))

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

#define PIPELINE_STAGES 2
constexpr unsigned int len = 10000U;
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
    typename GEMM::a_value_type, cublasdx::arrangement_of<GEMM>::a == cublasdx::arrangement::col_major, cublasdx::alignment_of<GEMM>::a,
    typename GEMM::b_value_type, cublasdx::arrangement_of<GEMM>::b == cublasdx::arrangement::col_major, cublasdx::alignment_of<GEMM>::b,
    typename GEMM::c_value_type, cublasdx::arrangement_of<GEMM>::c == cublasdx::arrangement::col_major, cublasdx::alignment_of<GEMM>::c,
    cublasdx::size_of<GEMM>::m, cublasdx::size_of<GEMM>::n, cublasdx::size_of<GEMM>::k>;
}

template<class BlockMM, typename MatrixA, typename MatrixB, typename MatrixC>
requires (cute::is_tensor_v<MatrixA> && cute::is_tensor_v<MatrixB> && cute::is_tensor_v<MatrixC>
    && cublasdx::is_complete_blas<BlockMM>::value
    && cublasdx::is_supported<BlockMM, cublasdx::sm_of<BlockMM>::value>::value
    && cublasdx::sm_of<BlockMM>::value >= MIN_ARCH)
__global__ void deviceCollectiveMMA(MatrixA mA, MatrixB mB, MatrixC mC) {
    static_assert(rank(mA) == 2 && rank(mB) == 2 && rank(mC) == 2);
    using Parameters = CollectiveMMAConfig<BlockMM>;
    constexpr auto bM = cublasdx::size_of<BlockMM>::m;
    constexpr auto bN = cublasdx::size_of<BlockMM>::n;
    constexpr auto bK = cublasdx::size_of<BlockMM>::k;
    using blockTiler = cute::Shape<cute::Int<bM>, cute::Int<bN>, cute::Int<bK>>;

    using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
        cutlass::gemm::MainloopSm80CpAsyncUnpredicated<PIPELINE_STAGES>,
        blockTiler,
        typename BlockMM::a_value_type,
        cute::Underscore,
        typename BlockMM::b_value_type,
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

    // Get the appropriate blocks for this thread block
    auto M = cute::get<0>(mC.shape());
    auto N = cute::get<1>(mC.shape());
    auto cta_coordX = cute::idx2crd(blockIdx.x, cute::Shape(M, N));
    auto cta_coord = cute::make_coord(cute::get<0>(cta_coordX), cute::get<1>(cta_coordX), cute::_);
    auto gA = local_tile(mA, blockTiler{}, cta_coord, cute::Step<cute::_1, cute::X,cute::_1>{});  // (BLK_M,BLK_K,k)
    auto gB = local_tile(mB, blockTiler{}, cta_coord, cute::Step< cute::X,cute::_1,cute::_1>{});  // (BLK_N,BLK_K,k)
    auto gC = local_tile(mC, blockTiler{}, cta_coord, cute::Step<cute::_1,cute::_1, cute::X>{});  // (BLK_M,BLK_N)

    auto k_tile_iter = cute::make_coord_iterator(size<2>(gA));
    int k_tile_count = size<2>(gA);

    extern __shared__ char buf[];
    CollectiveMainloop collective_mma;
    collective_mma(
        accum,
        gA,
        gB,
        accum,
        k_tile_iter, k_tile_count,
        cute::Underscore{},
        threadIdx.x,
        buf);
}

void testCollective() {
    const auto playStream = cudaStreamPerThread;
    constexpr auto M = 128;
    constexpr auto N = 64;
    constexpr auto K = 8;
    using inputValueType = cublasdx::tfloat32_t;
    using weightValueType = cublasdx::tfloat32_t;
    using outValueType = float;
    // Do y=xA^T
    using GEMM = decltype(cublasdx::Size<M, N, K>()
                          + cublasdx::Precision<inputValueType, weightValueType, outValueType>()
                          + cublasdx::Type<cublasdx::type::real>()
                          + cublasdx::Arrangement<cublasdx::row_major, cublasdx::row_major>()
                          + cublasdx::Function<cublasdx::function::MM>()
                          + cublasdx::SM<800>()
                          + cublasdx::Block());

    constexpr auto sharedSize = (sizeof(GEMM::a_value_type) * GEMM::a_size)
    + (sizeof(GEMM::b_value_type) * GEMM::b_size);
    constexpr auto abcSize = sharedSize + (sizeof(GEMM::b_value_type) * GEMM::c_size);
    cuda::std::byte* abc;
    CUTE_CHECK_ERROR(cudaMallocAsync(&abc, abcSize, playStream));

    int i = 0;
    static_assert(sizeof(inputValueType) == sizeof(weightValueType));
    static_assert(sizeof(inputValueType) == sizeof(outValueType));
    static_assert(sizeof(weightValueType) == sizeof(outValueType));
    auto* data = malloc(abcSize);
#pragma unroll
    for (;i < GEMM::a_size; ++i) {
        static_cast<inputValueType*>(data)[i] = static_cast<inputValueType>(0.01);
    }

#pragma unroll
    for (int j = 0; i < GEMM::a_size + GEMM::b_size; ++i, ++j) {
        static_cast<weightValueType*>(data)[i] = static_cast<weightValueType>(0.02);
    }
    CUTE_CHECK_ERROR(cudaMemcpyAsync(abc, data, abcSize, cudaMemcpyHostToDevice, playStream));

    auto mA = make_tensor(cute::make_gmem_ptr(CAST_TO(inputValueType, abc)), make_layout(cute::make_shape(M, K), cute::make_stride(K, 1)));
    auto mB = make_tensor(cute::make_gmem_ptr(CAST_TO(weightValueType, abc)), make_layout(cute::make_shape(N, K), cute::make_stride(K, 1)));
    auto mC = make_tensor(cute::make_gmem_ptr(CAST_TO(outValueType, abc)), make_layout(cute::make_shape(M, N), cute::make_stride(1, M)));

    deviceCollectiveMMA<GEMM><<<1, 128, sharedSize, playStream>>>(mA, mB, mC);
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

void debugLayout() {
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
}

__device__ cute::tfloat32_t matABC[11264];
template<class GEMM>
requires (cublasdx::is_complete_blas<GEMM>::value
    && cublasdx::is_supported<GEMM, cublasdx::sm_of<GEMM>::value>::value
    && cublasdx::sm_of<GEMM>::value >= MIN_ARCH)
__global__ void testCopy() {
    __shared__ cute::tfloat32_t sBuf[3072];
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
    auto gA = local_tile(mA, TileShape{}, cta_coord, cute::Step<cute::_1, cute::X,cute::_1>{});  // (BLK_M,BLK_K,k)
    auto gB = local_tile(mB, TileShape{}, cta_coord, cute::Step< cute::X,cute::_1,cute::_1>{});  // (BLK_N,BLK_K,k)
    auto gC = local_tile(mC, TileShape{}, cta_coord, cute::Step<cute::_1,cute::_1, cute::X>{});  // (BLK_M,BLK_N)

    using SmemLayoutA = decltype(tile_to_shape(
      typename Parameters::sLayA{},
      make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), PIPELINE_STAGES)));
    using SmemLayoutB = decltype(tile_to_shape(
        typename Parameters::sLayB{},
        make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}), PIPELINE_STAGES)));

    auto sA = make_tensor(cute::make_smem_ptr(sBuf), SmemLayoutA{}); // (BLK_M,BLK_K,PIPE)
    auto sB = make_tensor(cute::make_smem_ptr(sBuf + 2048), SmemLayoutB{}); // (BLK_N,BLK_K,PIPE)

    auto tAgA = gmem_thr_copy_A.partition_S(gA);                             // (ACPY,ACPY_M,ACPY_K,k)
    auto tAsA = gmem_thr_copy_A.partition_D(sA);                             // (ACPY,ACPY_M,ACPY_K,PIPE)
    auto tBgB = gmem_thr_copy_B.partition_S(gB);                             // (BCPY,BCPY_N,BCPY_K,k)
    auto tBsB = gmem_thr_copy_B.partition_D(sB);                             // (BCPY,BCPY_N,BCPY_K,PIPE)

    auto k_tile_iter = cute::make_coord_iterator(size<2>(gA));
    int k_tile_count = size<2>(gA);

    auto tiled_mma = cute::TiledMMA<
      cute::MMA_Atom<cute::SM80_16x8x8_F32TF32TF32F32_TN>,
      cute::Layout<cute::Shape<cute::_2, cute::_2, cute::_1>>,
    cute::Tile<cute::_32, cute::_32, cute::_8>
    >{};

    static_assert(cuda::std::is_same_v<decltype(tiled_mma), typename Parameters::mma_t>);

    auto smem_tiled_copy_A = make_tiled_copy_A(typename Parameters::sCopyA{}, tiled_mma);
    //auto smem_tiled_copy_A = make_tiled_copy_A(cute::Copy_Atom<cute::SM75_U32x4_LDSM_N, typename GEMM::a_value_type>{}, tiled_mma);
    auto smem_thr_copy_A   = smem_tiled_copy_A.get_thread_slice(threadIdx.x);
   auto smem_tiled_copy_B = make_tiled_copy_B(typename Parameters::sCopyB{}, tiled_mma);
    //auto smem_tiled_copy_B = make_tiled_copy_B(cute::Copy_Atom<cute::SM75_U32x4_LDSM_N, typename GEMM::a_value_type>{}, tiled_mma);
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
    copy(smem_tiled_copy_A, tCsA_p(cute::_,cute::_,0), tCrA_copy_view(cute::_,cute::_,0));
    copy(smem_tiled_copy_B, tCsB_p(cute::_,cute::_,0), tCrB_copy_view(cute::_,cute::_,0));
}
int main() {
    constexpr auto M = 128;
    constexpr auto N = 64;
    constexpr auto K = 8;
    using inputValueType = cublasdx::tfloat32_t;
    using weightValueType = cublasdx::tfloat32_t;
    using outValueType = float;
    // Has to be (M, K) * (N, K)
    using GEMM = decltype(cublasdx::Size<M, N, K>()
                          + cublasdx::Precision<inputValueType, weightValueType, outValueType>()
                          + cublasdx::Type<cublasdx::type::real>()
                          + cublasdx::Arrangement<cublasdx::row_major, cublasdx::row_major>()
                          + cublasdx::Function<cublasdx::function::MM>()
                          + cublasdx::SM<800>()
                          + cublasdx::Block());
    using Parameters = CollectiveMMAConfig<GEMM>;

    testCopy<GEMM><<<1,THREADS>>>();
    CUTE_CHECK_LAST();
}