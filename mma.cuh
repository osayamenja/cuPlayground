//
// Created by oja7 on 12/18/24.
//

#ifndef MMA_CUH
#define MMA_CUH

#include <cooperative_groups/memcpy_async.h>
#include <cuda/std/array>
#include <cuda/std/type_traits>
#include <cublasdx.hpp>
#include <cute/tensor.hpp>

#include "processor/tiling.cuh"
#include "util.cuh"

enum class ResultType {
    local,
    network
};

enum class ReleaseType {
    stable,
    experimental
};

#define MULTIPLE_TIMING 1
template<
    unsigned int M,
    unsigned int N,
    unsigned int K,
    class BlockGEMM,
    unsigned int sharedSize = 16 * 1024,
    ResultType r = ResultType::local,
    ReleaseType rT = ReleaseType::stable,
    typename MatrixA,
    typename MatrixB,
    typename MatrixC,
    typename MatrixD
>
requires (cute::is_tensor_v<MatrixA>
    && cute::is_tensor_v<MatrixB>
    && cute::is_tensor_v<MatrixC>
    && cute::is_tensor_v<MatrixD>
    && sharedSize % THREADS == 0)
__global__ __maxnreg__(255) void deviceCollectiveMMA(
    const MatrixA mA, const MatrixB mB, const MatrixC mC, const MatrixD mD, const bool skip = true) {
    using ElementD = typename decltype(mD)::value_type;
    static_assert(sharedSize % sizeof(ElementD) == 0);
    __shared__ __align__(16) ElementD scratch[sharedSize / sizeof(ElementD)];
#if MULTIPLE_TIMING
    float clocked = 0.0f;
    constexpr auto rounds = 8;
    for (uint k = 0; k < rounds; ++k) {
        uint64_t start = 0, end = 0;
        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(start)::);
#endif
        static_assert(rank(mA) == 2 && rank(mB) == 2 && rank(mC) == 2 && rank(mD) == 2);
        using ElementC = typename BlockGEMM::MatrixCType;
        constexpr auto bM = cute::get<0>(typename BlockGEMM::BlockTiler{});
        constexpr auto bN = cute::get<1>(typename BlockGEMM::BlockTiler{});
        constexpr auto bK = cute::get<2>(typename BlockGEMM::BlockTiler{});
        static_assert(M % bM == 0 && N % bN == 0 && K % bK == 0);
        constexpr typename BlockGEMM::MMA tiledMMA{};
        auto accum = cute::partition_fragment_C(tiledMMA, typename BlockGEMM::TilerOut{});
        static_assert(cuda::std::is_same_v<ElementC, typename decltype(accum)::value_type>);
        constexpr auto tM = M / bM;
        constexpr auto tN = N / bN;
        constexpr auto tK = K / bK;
        // blockIdx == tileIdx
        const auto cta_coordX = cute::idx2crd(blockIdx.x,
            cute::Shape<cute::Int<tM>, cute::Int<tN>>{},
                cute::Stride<cute::Int<tM>, cute::_1>{});
        const auto cta_coord = cute::make_coord(cute::get<0>(cta_coordX), cute::get<1>(cta_coordX), cute::_);
        const auto gA = local_tile(mA, typename BlockGEMM::BlockTiler{}, cta_coord, cute::Step<cute::_1, cute::X,cute::_1>{});  // (BLK_M,BLK_K,k)
        const auto gB = local_tile(mB, typename BlockGEMM::BlockTiler{}, cta_coord, cute::Step< cute::X,cute::_1,cute::_1>{});  // (BLK_N,BLK_K,k)
        const auto gC = local_tile(mC, typename BlockGEMM::BlockTiler{}, cta_coord, cute::Step<cute::_1,cute::_1, cute::X>{});  // (BLK_M,BLK_N)
        const auto gD = local_tile(mD, typename BlockGEMM::BlockTiler{}, cta_coord, cute::Step<cute::_1,cute::_1, cute::X>{});  // (BLK_M,BLK_N)
        const auto k_tile_iter = cute::make_coord_iterator(tK);

        static_assert(cuda::std::is_same_v<typename BlockGEMM::MatrixAType, typename BlockGEMM::MatrixBType> &&
            cuda::std::is_same_v<typename BlockGEMM::MatrixAType, ElementD>);
        cute::clear(accum);
        typename BlockGEMM::CollectiveMainloop collective_mma{};
        collective_mma(
            accum,
            gA,
            gB,
            accum,
            k_tile_iter, tK,
            cute::Underscore{},
            threadIdx.x,
            CAST_TO(char, scratch));

        // Epilogue
        //const auto tCgC = tiledMMA.get_slice(threadIdx.x).partition_C(gC);
        const auto tDgD = tiledMMA.get_slice(threadIdx.x).partition_C(gD);
        constexpr auto gDLoadOp = cutlass::NumericConverter<ElementC, ElementD>{};

        // Assume unary operator
        constexpr typename BlockGEMM::FusedEpilogue epilogueOp{};
        constexpr auto elems = bN;
        static_assert(size(accum) % elems == 0);
        constexpr auto trips = size(accum) / elems;
        // Leverage compiler packing for half-precision values into one register
        cutlass::AlignedArray<ElementD, elems> rScratch{};

        // vector type for vectorized copy
        using VT = uint32_t;
        // collectively copy from global to shared
        constexpr auto sCLay = cute::make_layout(cute::Shape<cute::Int<bM>, cute::Int<elems>>{}, cute::LayoutRight{});
        const auto sC = cute::make_tensor(cute::make_smem_ptr(scratch), sCLay);
        const auto rIdx = threadIdx.x / elems * elems;
        const auto cIdx = threadIdx.x % elems;
        const auto tCsC = tiledMMA.get_slice(threadIdx.x).partition_C(sC);
        const auto tCrC = cute::make_tensor(CAST_TO(ElementD, accum.data()),
            cute::Layout<cute::Shape<cute::Int<size(accum)>, cute::_1>>{});

        if constexpr (rT == ReleaseType::experimental) {
            constexpr auto vZ = sizeof(ElementD) / sizeof(VT);
            constexpr auto eVz = elems / vZ;
            // Data should be available in shared memory now
            #pragma unroll
            for (uint i = 0; i < trips; ++i) {
                // global -> shared
                #pragma unroll
                for (uint j = 0; j < elems; ++j) {
                    sC(rIdx + j, cIdx) = gD(rIdx + j, cIdx + i * elems);
                }
                __syncthreads();
                // shared -> registers
                #pragma unroll
                for (uint j = 0; j < eVz; ++j) {
                    CAST_TO(VT, rScratch.data())[j] = *CONST_CAST_TO(VT, &tCsC(j * vZ));
                }
                #pragma unroll
                for (int j = 0; j < elems; ++j) {
                    accum(j + i * elems) = epilogueOp(accum(j + i * elems), gDLoadOp(rScratch[j]));
                }
            }
        }
        else {
            using LT =  cuda::std::conditional_t<sizeof(ElementD) == 2, uint16_t, uint32_t>;
            // Prefetch from global to shared memory with vector accesses
            #pragma unroll
            for (int j = 0; j < elems; ++j) {
                CAST_TO(LT, scratch)[threadIdx.x + j * THREADS] = __ldg(CONST_CAST_TO(LT, &tDgD(j)));
            }

            #pragma unroll
            for (unsigned int i = 0; i < trips; ++i) {
                #pragma unroll
                for (unsigned int j = 0; j < elems; ++j) {
                    rScratch[j] = scratch[threadIdx.x + j * THREADS];
                    if (i + 1 < trips) {
                        // Eagerly start loads for the next batch, if needed
                        scratch[threadIdx.x + j * THREADS] = tDgD(j + (i + 1) * elems);
                    }
                }
                // Fused Bias Add and Activation Function on register fragment
                #pragma unroll
                for (int j = 0; j < elems; ++j) {
                    accum(j + i * elems) = epilogueOp(accum(j + i * elems), gDLoadOp(rScratch[j]));
                }
            }
        }

        __syncthreads();
        constexpr auto gCStoreOp = cutlass::NumericConverter<typename decltype(gC)::value_type, ElementC>{};
        #pragma unroll
        for (unsigned int i = 0; i < trips; ++i) {
            #pragma unroll
            for (unsigned int j = 0; j < elems; ++j) {
                tCsC(j) = gCStoreOp(accum(j + i * elems));
            }
            __syncthreads();
            #pragma unroll
            for (unsigned int j = 0; j < elems; ++j) {
                gC(rIdx + j, cIdx + i * elems) = sC(rIdx + j, cIdx);
            }
        }
        __syncthreads();
        if (!threadIdx.x) {
            if constexpr (r == ResultType::local) {
                __threadfence();
            }
            else {
                __threadfence_system();
            }
        }
        __syncthreads();
#if MULTIPLE_TIMING
        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(end)::);
        clocked += static_cast<float>(end - start) / static_cast<float>(rounds);
    }
    if (!skip && !threadIdx.x) {
        printf("Block %u, Duration was %fus\n", blockIdx.x, clocked / 1000.0f);
    }
#endif

#if 0
    if (!threadIdx.x && !skip) {
        /*cute::print_tensor(mD);
        cute::print_tensor(mA);
        cute::print_tensor(mB);*/
        cute::print_tensor(mC);
    }
#endif
}

__host__ __forceinline__
void testCollective() {
    const auto playStream = cudaStreamPerThread;
    constexpr auto M = 1280;
    constexpr auto N = 1280;
    constexpr auto K = 4096;

    using inputValueType = cute::half_t;
    using outValueType = inputValueType;
    using weightValueType = inputValueType;
    using accumulateType = float;

    using activation = cutlass::epilogue::thread::ReLU<accumulateType>;
    using Operation = BlockMM<800, inputValueType, weightValueType, accumulateType, activation>;
    constexpr auto aSize = (sizeof(inputValueType) * M * K);
    constexpr auto abSize = aSize + (sizeof(weightValueType) * N * K);
    constexpr auto abcSize = abSize + (sizeof(outValueType) * M * N);
    constexpr auto len = abcSize + (sizeof(inputValueType) * cute::max(N, K));
    cuda::std::byte* abc;
    CHECK_ERROR_EXIT(cudaMallocAsync(&abc, len, playStream));
    CHECK_ERROR_EXIT(cudaMemsetAsync(abc, 0, len, playStream));
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

    CHECK_ERROR_EXIT(cudaMemcpyAsync(abc, data, len, cudaMemcpyHostToDevice, playStream));

    const auto mA = make_tensor(cute::make_gmem_ptr(CONST_CAST_TO(inputValueType, abc)),
        cute::Layout<cute::Shape<cute::Int<M>, cute::Int<K>>,cute::Stride<cute::Int<K>, cute::_1>>{});
    const auto mB = make_tensor(cute::make_gmem_ptr(
        CONST_CAST_TO(weightValueType, abc + aSize)),
        cute::Layout<cute::Shape<cute::Int<N>, cute::Int<K>>,cute::Stride<cute::Int<K>, cute::_1>>{});
    const auto mC = make_tensor(cute::make_gmem_ptr(CAST_TO(inputValueType, abc + abSize)),
        cute::Layout<cute::Shape<cute::Int<M>, cute::Int<N>>,cute::Stride<cute::Int<N>, cute::_1>>{});

    // bias vector (1, N) broadcast to (M, N)
    const auto mD = make_tensor(cute::make_gmem_ptr(CONST_CAST_TO(inputValueType, abc + abcSize)),
        cute::Layout<cute::Shape<cute::Int<M>, cute::Int<N>>,cute::Stride<cute::_0, cute::_1>>{});

    constexpr auto gemmSharedSize = sizeof(inputValueType) * Operation::GEMM::a_size
        + (sizeof(weightValueType) * Operation::GEMM::b_size);
    constexpr auto sharedSize = cute::max(gemmSharedSize * PIPELINE_STAGES, 48 * 1024);
    constexpr auto blocks = (M / cute::get<0>(Operation::BlockTiler{})) * (N / cute::get<1>(Operation::BlockTiler{}));
    #if MULTIPLE_TIMING
    for (uint i = 0; i < 128; ++i) {
        deviceCollectiveMMA<M, N, K, Operation, sharedSize><<<blocks, 128, 0, playStream>>>(mA, mB, mC, mD);
    }
    #endif
    deviceCollectiveMMA<M, N, K, Operation, sharedSize, ResultType::local>
    <<<blocks, 128, 0, playStream>>>(mA, mB, mC, mD, false);
    CHECK_LAST();
    CHECK_ERROR_EXIT(cudaFree(abc));
    free(data);
}

/*__host__ __forceinline__
void testP2PCollective() {
    CHECK_ERROR_EXIT(cudaSetDevice(0));
    CHECK_ERROR_EXIT(cudaDeviceEnablePeerAccess(1, 0));
    // use peer memory for transfer
    const auto playStream = cudaStreamPerThread;
    constexpr auto M = 128;
    constexpr auto N = 64;
    constexpr auto K = 1024;

    using inputValueType = cute::half_t;
    using outValueType = inputValueType;
    using weightValueType = cute::half_t;
    using accumulateType = float;

    using activation = cutlass::epilogue::thread::ReLU<accumulateType>;
    using Operation = BlockMM<800, inputValueType, weightValueType, accumulateType, activation>;
    constexpr auto aSize = sizeof(inputValueType) * M * K;
    constexpr auto abSize = aSize + sizeof(weightValueType) * N * K;
    constexpr auto cSize = sizeof(outValueType) * M * N;
    constexpr auto len = abSize + sizeof(inputValueType) * cute::max(N, K);
    cuda::std::byte* abc;
    CHECK_ERROR_EXIT(cudaMallocAsync(&abc, len, playStream));
    CHECK_ERROR_EXIT(cudaMemsetAsync(abc, 0, len, playStream));

    cuda::std::byte* peerMemory;
    // Switch to PE 1
    CHECK_ERROR_EXIT(cudaSetDevice(1));
    CHECK_ERROR_EXIT(cudaDeviceEnablePeerAccess(0, 0));
    CHECK_ERROR_EXIT(cudaMalloc(&peerMemory, cSize));
    CHECK_ERROR_EXIT(cudaPeekAtLastError());
    // Switch back to PE 0
    CHECK_ERROR_EXIT(cudaSetDevice(0));
    auto* data = static_cast<cuda::std::byte*>(calloc(len, sizeof(cuda::std::byte)));

    const auto mAHost = make_tensor(CAST_TO(inputValueType, data),
        make_layout(cute::make_shape(M, K), cute::make_stride(K, 1)));
    const auto mBHost = make_tensor(CAST_TO(weightValueType, data + aSize),
        make_layout(cute::make_shape(N, K), cute::make_stride(K, 1)));

    // Populate bias vector
    CAST_TO(inputValueType, data + abSize)[0] = static_cast<inputValueType>(1.0);
    CAST_TO(inputValueType, data + abSize)[1] = static_cast<inputValueType>(2.0);

    mAHost(0, 0) = static_cast<inputValueType>(0.0);
    mAHost(0, 1) = static_cast<inputValueType>(1.0);
    mAHost(1, 0) = static_cast<inputValueType>(2.0);
    mAHost(1, 1) = static_cast<inputValueType>(3.0);

    mBHost(0, 0) = static_cast<weightValueType>(4.0);
    mBHost(0, 1) = static_cast<weightValueType>(5.0);
    mBHost(1, 0) = static_cast<weightValueType>(6.0);
    mBHost(1, 1) = static_cast<weightValueType>(7.0);

    CHECK_ERROR_EXIT(cudaMemcpyAsync(abc, data, len, cudaMemcpyHostToDevice, playStream));

    const auto mA = make_tensor(cute::make_gmem_ptr(CAST_TO(inputValueType, abc)),
        make_layout(cute::make_shape(M, K), cute::make_stride(K, 1)));
    const auto mB = make_tensor(cute::make_gmem_ptr(
        CAST_TO(weightValueType, abc + aSize)), make_layout(cute::make_shape(N, K), cute::make_stride(K, 1)));
    // This is peer memory
    const auto mC = make_tensor(cute::make_gmem_ptr(CAST_TO(inputValueType, peerMemory)),
        make_layout(cute::make_shape(M, N), cute::make_stride(N, 1)));

    // bias vector (1, N) broadcast to (M, N)
    const auto mD = make_tensor(cute::make_gmem_ptr(CAST_TO(inputValueType, abc + abSize)),
        make_layout(cute::make_shape(M, N), cute::make_stride(0, 1)));

    constexpr auto gemmSharedSize = sizeof(inputValueType) * Operation::GEMM::a_size
        + (sizeof(weightValueType) * Operation::GEMM::b_size);
    constexpr auto sharedSize = cute::max(gemmSharedSize * PIPELINE_STAGES, 128 * 64 * 4);
#if MULTIPLE_TIMING
    for (uint i = 0; i < 128; ++i) {
        deviceCollectiveMMA<M, N, K, Operation, sharedSize, ResultType::network><<<1, 128, 0, playStream>>>
        (mA, mB, mC, mD);
    }
#endif
    deviceCollectiveMMA<M, N, K, Operation, sharedSize, ResultType::network><<<1, 128, 0, playStream>>>
        (mA, mB, mC, mD, false);
    CHECK_ERROR_EXIT(cudaPeekAtLastError());
    // wait for kernel to complete
    CHECK_ERROR_EXIT(cudaStreamSynchronize(playStream));

    CHECK_ERROR_EXIT(cudaFreeAsync(abc, playStream));

    CHECK_ERROR_EXIT(cudaSetDevice(1));
    CHECK_ERROR_EXIT(cudaFree(peerMemory));
    CHECK_ERROR_EXIT(cudaPeekAtLastError());
    CHECK_ERROR_EXIT(cudaDeviceSynchronize());

    CHECK_ERROR_EXIT(cudaSetDevice(0));
    CHECK_ERROR_EXIT(cudaPeekAtLastError());
    CHECK_ERROR_EXIT(cudaStreamSynchronize(playStream));
    free(data);
}*/

#endif //MMA_CUH
