//
// Created by oja7 on 12/18/24.
//

#ifndef MMA_CUH
#define MMA_CUH

#include <cuda/std/type_traits>
#include <cublasdx.hpp>
#include <cute/tensor.hpp>

#include "processor/tiling.cuh"
#include "util.cuh"


template<class BlockGEMM, unsigned int sharedSize = 16 * 1024,
typename MatrixA, typename MatrixB, typename MatrixC, typename MatrixD>
requires (cute::is_tensor_v<MatrixA>
    && cute::is_tensor_v<MatrixB>
    && cute::is_tensor_v<MatrixC>
    && cute::is_tensor_v<MatrixD>
    && sharedSize % THREADS == 0)
__global__ __maxnreg__(128) void deviceCollectiveMMA(
    const MatrixA mA, const MatrixB mB, const MatrixC mC, const MatrixD mD) {
    static_assert(rank(mA) == 2 && rank(mB) == 2 && rank(mC) == 2 && rank(mD) == 2);
    using ElementC = typename BlockGEMM::MatrixCType;
    constexpr auto bM = cute::get<0>(typename BlockGEMM::BlockTiler{});
    constexpr auto bN = cute::get<0>(typename BlockGEMM::BlockTiler{});

    constexpr typename BlockGEMM::MMA tiledMMA{};
    auto accum = cute::partition_fragment_C(tiledMMA, typename BlockGEMM::TilerOut{});
    static_assert(cuda::std::is_same_v<ElementC, typename decltype(accum)::value_type>);
    cute::clear(accum);

    // Get the appropriate blocks for this thread block
    // use problem shape instead, p_MNK = (cute::ceil_div(M, bM), cute::ceil_div(N, bN), K)
    //auto M = cute::get<0>(mC.shape());
    //auto N = cute::get<1>(mC.shape());
    const auto cta_coordX = cute::idx2crd(blockIdx.x, cute::Shape(cute::ceil_div(cute::get<0>(mC.shape()), bM),
        cute::ceil_div(cute::get<1>(mC.shape()), bN)));
    const auto cta_coord = cute::make_coord(cute::get<0>(cta_coordX), cute::get<1>(cta_coordX), cute::_);
    const auto gA = local_tile(mA, typename BlockGEMM::BlockTiler{}, cta_coord, cute::Step<cute::_1, cute::X,cute::_1>{});  // (BLK_M,BLK_K,k)
    const auto gB = local_tile(mB, typename BlockGEMM::BlockTiler{}, cta_coord, cute::Step< cute::X,cute::_1,cute::_1>{});  // (BLK_N,BLK_K,k)
    const auto gC = local_tile(mC, typename BlockGEMM::BlockTiler{}, cta_coord, cute::Step<cute::_1,cute::_1, cute::X>{});  // (BLK_M,BLK_N)
    const auto gD = local_tile(mD, typename BlockGEMM::BlockTiler{}, cta_coord, cute::Step<cute::_1,cute::_1, cute::X>{});  // (BLK_M,BLK_N)

    auto k_tile_iter = cute::make_coord_iterator(size<2>(gA));
    int k_tile_count = size<2>(gA);

    using ElementD = typename decltype(gD)::value_type;
    extern __shared__ ElementD scratch[];
    typename BlockGEMM::CollectiveMainloop collective_mma;
    collective_mma(
        accum,
        gA,
        gB,
        accum,
        k_tile_iter, k_tile_count,
        cute::Underscore{},
        threadIdx.x,
        CAST_TO(char, scratch));

    // Epilogue

    const auto tCgC = tiledMMA.get_slice(threadIdx.x).partition_C(gC);
    const auto tDgD = tiledMMA.get_slice(threadIdx.x).partition_C(gD);
    
    // Accounts for GEMMs that accumulate in types differing from input types,
    // given that the result moonlights as the input for the succeeding GEMM.
    constexpr auto gCStoreOp = cutlass::NumericConverter<typename decltype(tCgC)::value_type, ElementC>{};
    constexpr auto gDLoadOp = cutlass::NumericConverter<ElementC, ElementD>{};

    // Assume unary operator
    constexpr typename BlockGEMM::FusedEpilogue epilogueOp{};
    constexpr auto elems = sharedSize / (THREADS * sizeof(ElementD));
    static_assert(size(accum) % elems == 0);
    constexpr auto trips = size(accum) / elems;
    // Leverage compiler packing for half-precision values into one register
    cutlass::AlignedArray<ElementD, elems> rScratch{};

    // Prefetch from global to shared memory
    #pragma unroll
    for (int j = 0; j < elems; ++j) {
        scratch[threadIdx.x + j * THREADS] = tDgD(j);
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
        // Also fuses copy to GMEM.
        #pragma unroll
        for (int j = 0; j < elems; ++j) {
            tCgC(j + i * elems) = gCStoreOp(epilogueOp(accum(j + i * elems), gDLoadOp(rScratch[j])));
        }
    }

    __syncthreads();
    if (!threadIdx.x) {
        __threadfence();
    }
    __syncthreads();

#if 1
    if (!threadIdx.x) {
        cute::print_tensor(mD);
        cute::print_tensor(mA);
        cute::print_tensor(mB);
        cute::print_tensor(mC);
    }
#endif
}

__host__ __forceinline__
void testCollective() {
    const auto playStream = cudaStreamPerThread;
    constexpr auto M = 128;
    constexpr auto N = 64;
    constexpr auto K = 64;

    using inputValueType = cute::half_t;
    using outValueType = inputValueType;
    using weightValueType = cute::half_t;
    using accumulateType = float;

    using activation = cutlass::epilogue::thread::ReLU<accumulateType>;
    using Operation = BlockMM<inputValueType, weightValueType, accumulateType, 800, activation>;
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

    const auto mA = make_tensor(cute::make_gmem_ptr(CAST_TO(inputValueType, abc)),
        make_layout(cute::make_shape(M, K), cute::make_stride(K, 1)));
    const auto mB = make_tensor(cute::make_gmem_ptr(
        CAST_TO(weightValueType, abc + aSize)), make_layout(cute::make_shape(N, K), cute::make_stride(K, 1)));
    const auto mC = make_tensor(cute::make_gmem_ptr(CAST_TO(inputValueType, abc + abSize)),
        make_layout(cute::make_shape(M, N), cute::make_stride(N, 1)));

    // bias vector (1, N) broadcast to (M, N)
    const auto mD = make_tensor(cute::make_gmem_ptr(CAST_TO(inputValueType, abc + abcSize)),
        make_layout(cute::make_shape(M, N), cute::make_stride(0, 1)));

    constexpr auto gemmSharedSize = (sizeof(inputValueType) * Operation::GEMM::a_size)
        + (sizeof(weightValueType) + Operation::GEMM::b_size);
    constexpr auto sharedSize = cute::max(gemmSharedSize * PIPELINE_STAGES, 128 * 32 * 4);
    deviceCollectiveMMA<Operation, sharedSize><<<1, 128, sharedSize, playStream>>>(mA, mB, mC, mD);
    CUTE_CHECK_LAST();
    CUTE_CHECK_ERROR(cudaFree(abc));
    free(data);
}

#endif //MMA_CUH
