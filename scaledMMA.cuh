//
// Created by oja7 on 12/24/24.
//

#ifndef SCALEDMMA_CUH
#define SCALEDMMA_CUH
#include <cuda/std/type_traits>
#include <cublasdx.hpp>
#include <cute/tensor.hpp>
#include <cutlass/epilogue/thread/activation.h>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/gemm/collective/collective_mma.hpp>

#include "processor/tiling.cuh"
#include "util.cuh"

template<class BlockMM, typename ActivationOp = cute::identity,
unsigned int sharedSize = 16 * 1024,
typename MatrixA, typename MatrixB, typename MatrixC, typename MatrixD, typename MatrixS,
typename MatrixAx, typename MatrixBx, typename MatrixCx, typename MatrixDx,
typename ElementA = toCT<typename BlockMM::a_value_type>,
typename ElementB = toCT<typename BlockMM::b_value_type>,
typename ElementC = toCT<typename BlockMM::c_value_type>>
requires (cute::is_tensor_v<MatrixA>
    && cute::is_tensor_v<MatrixB>
    && cute::is_tensor_v<MatrixC>
    && cute::is_tensor_v<MatrixD>
    && cute::is_tensor_v<MatrixS>
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
    const MatrixA mA, const MatrixB mB, const MatrixC mC, const MatrixD mD, const MatrixS mS,
    const MatrixAx mAx, const MatrixBx mBx, const MatrixCx mCx, const MatrixDx mDx) {
    static_assert(rank(mA) == 2 && rank(mB) == 2 && rank(mC) == 2 && rank(mD) == 2);
    static_assert(rank(mAx) == 2 && rank(mBx) == 2 && rank(mCx) == 2 && rank(mDx) == 2);
    using Parameters = CollectiveMMAConfig<BlockMM, LayoutOptimization::UseSwizzle>;
    constexpr auto bM = cublasdx::size_of<BlockMM>::m;
    constexpr auto bN = cublasdx::size_of<BlockMM>::n;
    constexpr auto bK = cublasdx::size_of<BlockMM>::k;
    using blockTiler = cute::Shape<cute::Int<bM>, cute::Int<bN>, cute::Int<bK>>;

    typename Parameters::mma_t tiledMMA;
    using TilerOut = cute::Shape<cute::Int<bM>, cute::Int<bN>>;
    auto accum = cute::partition_fragment_C(tiledMMA, TilerOut{});
    static_assert(cuda::std::is_same_v<ElementC, typename decltype(accum)::value_type>);
    cute::clear(accum);

    // Get the appropriate blocks for this thread block
    // use problem shape instead, p_MNK = (cute::ceil_div(M, bM), cute::ceil_div(N, bN), K)
    //auto M = cute::get<0>(mC.shape());
    //auto N = cute::get<1>(mC.shape());
    const auto cta_coordX = cute::idx2crd(blockIdx.x, cute::Shape(cute::ceil_div(cute::get<0>(mC.shape()), bM),
        cute::ceil_div(cute::get<1>(mC.shape()), bN)));
    const auto cta_coord = cute::make_coord(cute::get<0>(cta_coordX), cute::get<1>(cta_coordX), cute::_);
    const auto gA = local_tile(mA, blockTiler{}, cta_coord, cute::Step<cute::_1, cute::X,cute::_1>{});  // (BLK_M,BLK_K,k)
    const auto gB = local_tile(mB, blockTiler{}, cta_coord, cute::Step< cute::X,cute::_1,cute::_1>{});  // (BLK_N,BLK_K,k)
    const auto gC = local_tile(mC, blockTiler{}, cta_coord, cute::Step<cute::_1,cute::_1, cute::X>{});  // (BLK_M,BLK_N)
    const auto gD = local_tile(mD, blockTiler{}, cta_coord, cute::Step<cute::_1,cute::_1, cute::X>{});  // (BLK_M,BLK_N)
    const auto gS = local_tile(mS, blockTiler{}, cta_coord, cute::Step<cute::_1,cute::_1, cute::X>{});

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

    // Epilogue

    const auto tCgC = tiledMMA.get_slice(threadIdx.x).partition_C(gC);
    const auto tDgD = tiledMMA.get_slice(threadIdx.x).partition_C(gD);
    const auto tSgS = tiledMMA.get_slice(threadIdx.x).partition_C(gS);

    // Accounts for GEMMs that accumulate in types differing from input types,
    // given that the result moonlights as the input for the succeeding GEMM.
    constexpr auto gCStoreOp = cutlass::NumericConverter<typename decltype(tCgC)::value_type, ElementC>{};
    constexpr auto gDLoadOp = cutlass::NumericConverter<ElementC, ElementD>{};
    constexpr auto ScaleOp = cutlass::epilogue::thread::Scale<ElementC>{};

    // Assume unary operator
    ActivationOp epilogueOp{};
    constexpr auto elems = sharedSize / (THREADS * sizeof(ElementD));
    static_assert(size(accum) % elems == 0);
    constexpr auto trips = size(accum) / elems;
    // Leverage compiler packing for half-precision values into one register
    cutlass::AlignedArray<ElementD, elems> rScratch{};

    #pragma unroll
    for (int i = 0; i < trips; ++i) {
        // Prefetch from global to shared memory
        #pragma unroll
        for (int j = 0; j < elems; ++j) {
            scratch[threadIdx.x + j * THREADS] = tDgD(j);
        }

        #pragma unroll
        for (int j = 0; j < elems; ++j) {
            rScratch[j] = scratch[threadIdx.x + j * THREADS];
            /*if (i + 1 < trips) {
                // Eagerly start loads for the next batch
                scratch[threadIdx.x + j * THREADS] = tDgD(j + (i + 1) * elems);
            }*/
            scratch[threadIdx.x + j * THREADS] = tSgS(j + i * elems);
        }

        // Fused Bias Add and Activation Function on register fragment
        // Also fuses copy to GMEM
        #pragma unroll
        for (int j = 0; j < elems; ++j) {
            accum(j + i * elems) = gCStoreOp(fusedAddActivate(accum(j + i * elems),
                gDLoadOp(rScratch[j]), epilogueOp));
            rScratch[j] = scratch[threadIdx.x + j * THREADS];
        }

        // Do scale
        #pragma unroll
        for (int j = 0; j < elems; ++j) {
            tCgC(j + i * elems) = ScaleOp(accum(j + i * elems), gDLoadOp(rScratch[j]));
        }
    }

    __syncthreads();
    if (!threadIdx.x) {
        __threadfence();
    }
}

__host__ __forceinline__
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

    CHECK_ERROR_EXIT(cudaMemcpyAsync(abc, data, len, cudaMemcpyHostToDevice, playStream));

    const auto mA = make_tensor(cute::make_gmem_ptr(CAST_TO(inputValueType, abc)),
        make_layout(cute::make_shape(M, K), cute::make_stride(K, 1)));
    const auto mB = make_tensor(cute::make_gmem_ptr(
        CAST_TO(weightValueType, abc + aSize)), make_layout(cute::make_shape(N, K), cute::make_stride(K, 1)));
    const auto mC = make_tensor(cute::make_gmem_ptr(CAST_TO(inputValueType, abc + abSize)),
        make_layout(cute::make_shape(M, N), cute::make_stride(N, 1)));

    // bias vector (1, N) broadcast to (M, N)
    const auto mD = make_tensor(cute::make_gmem_ptr(CAST_TO(inputValueType, abc + abcSize)),
        make_layout(cute::make_shape(M, N), cute::make_stride(0, 1)));
    const auto mS = make_tensor(cute::make_gmem_ptr(CAST_TO(inputValueType, abc + abcSize)),
                make_layout(cute::make_shape(M, N), cute::make_stride(1, 0)));

    // Second GEMM
    const auto mAx = make_tensor(cute::make_gmem_ptr(CAST_TO(inputValueType, abc + abSize)),
        make_layout(cute::make_shape(M, N), cute::make_stride(N, 1)));
    const auto mBx = make_tensor(cute::make_gmem_ptr(
        CAST_TO(weightValueType, abc + aSize)), make_layout(cute::make_shape(K, N), cute::make_stride(N, 1)));
    const auto mCx = make_tensor(cute::make_gmem_ptr(CAST_TO(outValueType, abc)),
        make_layout(cute::make_shape(M, K), cute::make_stride(K, 1)));

    const auto mDx = make_tensor(cute::make_gmem_ptr(CAST_TO(inputValueType, abc + abcSize)),
        make_layout(cute::make_shape(M, K), cute::make_stride(0, 1)));

    constexpr auto gemmSharedSize = (sizeof(inputValueType) * GEMM::a_size)
        + (sizeof(weightValueType) + GEMM::b_size);
    constexpr auto sharedSize = cute::max(gemmSharedSize * PIPELINE_STAGES, 128 * 32 * 4);
    using activation = cutlass::epilogue::thread::ReLU<accumulateType>;
    deviceCollectiveMMA<GEMM, activation><<<1, 128, sharedSize, playStream>>>
    (mA, mB, mC, mD, mS, mAx, mBx, mCx, mDx);
    CHECK_LAST();
    CHECK_ERROR_EXIT(cudaFree(abc));
    free(data);
}
#endif //SCALEDMMA_CUH
