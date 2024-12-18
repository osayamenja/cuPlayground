//
// Created by oja7 on 12/18/24.
//

#ifndef MMA_CUH
#define MMA_CUH

#include <cuda/std/type_traits>
#include <cublasdx.hpp>
#include <cutlass/epilogue/thread/activation.h>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/gemm/collective/collective_mma.hpp>
#include "processor/tiling.cuh"
#include "util.cuh"

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

template<typename Acc>
requires (cute::is_tensor_v<Acc> && cute::is_rmem_v<Acc>)
using golf = cute::Int<2>;
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
    constexpr auto gg = golf<decltype(accum)>::value;
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

#endif //MMA_CUH
