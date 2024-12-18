//
// Created by oja7 on 12/18/24.
//

#ifndef DEBUG_CUH
#define DEBUG_CUH

#include <cute/tensor.hpp>
#include <cuda/std/type_traits>
#include <cuda/std/array>
#include <cublasdx.hpp>
#include "processor/tiling.cuh"
#include "processor/gemm.cuh"
#include <nvshmemx.h>
#include <nvshmem.h>
#include <host/nvshmemx_api.h> // Makes CLion happy
#include "util.cuh"

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

__constant__ __inline__ Args b{};
void __global__ testArgs() {
    printf("Args has rank %u, results %f\n", b.rank, b.result[0]);
    b.sHeap[0] = 45.0;
    b.result[0] = 59.0;
    printf("Args has rank %u, results %f\n", b.rank, b.result[0]);
}

__host__ __inline__
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

__host__ __inline__
void testGEMM() {
    introduction_example<800>();
}

__host__ __inline__
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

void mmaGolfing() {
    constexpr auto mma = cute::TiledMMA<
          cute::MMA_Atom<cute::SM80_16x8x8_F32F16F16F32_TN>,
          cute::Layout<cute::Shape<cute::_2, cute::_2, cute::_1>>,
        cute::Tile<cute::_64, cute::_32, cute::_8>
        >{};
    print_latex(mma);
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
#endif //DEBUG_CUH
