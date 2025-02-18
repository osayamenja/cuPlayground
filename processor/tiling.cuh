//
// Created by osayamen on 10/24/24.
//

#ifndef TILING_CUH
#define TILING_CUH

#include <cublasdx.hpp>
#include <cute/arch/copy.hpp>
#include <cute/arch/copy_sm80.hpp>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/gemm/collective/builders/sm90_common.inl>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_mma.hpp>
#include <cutlass/epilogue/thread/activation.h>

#include "../util.cuh"
// GEMM configuration constants
#define MIN_ARCH 700U
#define THREADS 128U
#define BLOCK_M 128U
#define BLOCK_N 64U
#define BLOCK_K_HALF 16U
#define BLOCK_K_FULL 8U
#define MAX_REGS (BLOCK_M * BLOCK_N) / THREADS
#define PIPELINE_STAGES 2U

/// Fused, Add, Activate
template <typename Element, typename ActivationFunction>
requires(TensorValueType<Element> && cuda::std::is_invocable_r_v<Element, ActivationFunction, Element>)
struct FAA {
    // fp8
    __forceinline__ __device__
    Element operator()(const Element& accumulator, const Element& term) const {
        constexpr ActivationFunction op{};
        return op(accumulator + term);
    }
};

// specialization for half-precision and relu
template<>
struct FAA<cute::half_t, cutlass::epilogue::thread::ReLU<cute::half_t>> {
    __forceinline__ __device__
    cute::half_t operator()(const cute::half_t& accumulator, const cute::half_t& term) const {
        return cute::half_t(__hfma_relu(__half(1.0f),accumulator.to_half(), term.to_half()));
    }
};

// specialization for bfloat16 and relu
template<>
struct FAA<cute::bfloat16_t, cutlass::epilogue::thread::ReLU<cute::bfloat16_t>> {
    __forceinline__ __device__
    cute::bfloat16_t operator()(const cute::bfloat16_t& accumulator, const cute::bfloat16_t& term) const {
        return cute::bfloat16_t(__hfma_relu(__nv_bfloat16(1.0f),
            accumulator.to_nv_bfloat16(), term.to_nv_bfloat16()));
    }
};

template<typename F>
struct isFAA : cuda::std::false_type {};

template<typename Element, typename ActivationFunction>
struct isFAA<FAA<Element, ActivationFunction>> : cuda::std::true_type {};

template<typename T>
using toCDX = cuda::std::conditional_t< cuda::std::is_same_v<T, cute::half_t>,
        __half,
    cuda::std::conditional_t<cuda::std::is_same_v<T, cute::bfloat16_t>,
        __nv_bfloat16,
    cuda::std::conditional_t<cuda::std::is_same_v<T, cute::float_e4m3_t>,
        __nv_fp8_e4m3,
    cuda::std::conditional_t<cuda::std::is_same_v<T, cute::float_e5m2_t>,
        __nv_fp8_e5m2, T>>>>;

template<typename T>
using toCT = cuda::std::conditional_t<cuda::std::is_same_v<T, __half>,
        cute::half_t,
    cuda::std::conditional_t<cuda::std::is_same_v<T, __nv_bfloat16>,
        cute::bfloat16_t,
    cuda::std::conditional_t<cuda::std::is_same_v<T, __nv_fp8_e4m3>,
        cute::float_e4m3_t,
    cuda::std::conditional_t<cuda::std::is_same_v<T, __nv_fp8_e5m2>,
        cute::float_e5m2_t, T>>>>;

template<unsigned int Arch, typename TC, typename TA=TC, typename TB=TA>
struct MMAConfig {
    using mma = cute::TiledMMA<
                cute::MMA_Atom<cute::UniversalFMA<TC, TA, TB>>,
                cute::Layout<cute::Shape<cute::_16, cute::_8, cute::_1>>
    >;
};

template<>
struct MMAConfig<700, cute::half_t> {
    using mma = cute::TiledMMA<
      cute::MMA_Atom<cute::SM70_8x8x4_F16F16F16F16_TN>,
      cute::Layout<cute::Shape<cute::_4, cute::_4, cute::_1>>,
    cute::Tile<cute::_32, cute::_32, cute::_8>
    >;
};

template<>
struct MMAConfig<700, float, cute::half_t> {
    using mma = cute::TiledMMA<
      cute::MMA_Atom<cute::SM70_8x8x4_F32F16F16F32_TN>,
      cute::Layout<cute::Shape<cute::_4, cute::_4, cute::_1>>,
    cute::Tile<cute::_32, cute::_32, cute::_8>
    >;
};

template<>
struct MMAConfig<800, cute::half_t> {
    using mma = cute::TiledMMA<
      cute::MMA_Atom<cute::SM80_16x8x8_F16F16F16F16_TN>,
      cute::Layout<cute::Shape<cute::_2, cute::_2, cute::_1>>,
    cute::Tile<cute::_32, cute::_32, cute::_8>
    >;
};

template<>
struct MMAConfig<800, float, cute::half_t> {
    using mma = cute::TiledMMA<
      cute::MMA_Atom<cute::SM80_16x8x8_F32F16F16F32_TN>,
      cute::Layout<cute::Shape<cute::_2, cute::_2, cute::_1>>,
    cute::Tile<cute::_32, cute::_32, cute::_8>
    >;
};

template<>
struct MMAConfig<800, float, cute::bfloat16_t> {
    using mma = cute::TiledMMA<
      cute::MMA_Atom<cute::SM80_16x8x8_F32BF16BF16F32_TN>,
      cute::Layout<cute::Shape<cute::_2, cute::_2, cute::_1>>,
    cute::Tile<cute::_32, cute::_32, cute::_8>
    >;
};

template<>
struct MMAConfig<800, float, cute::tfloat32_t> {
    using mma = cute::TiledMMA<
      cute::MMA_Atom<cute::SM80_16x8x8_F32TF32TF32F32_TN>,
      cute::Layout<cute::Shape<cute::_2, cute::_2, cute::_1>,
                    cute::Stride<cute::_2, cute::_1, cute::_1>>,
    cute::Tile<cute::_32, cute::_32, cute::_8>
    >;
};

template <cublasdx::arrangement a, unsigned int midSwizzle, unsigned int sizeK>
requires((a == cublasdx::arrangement::row_major || a == cublasdx::arrangement::col_major)
    && (midSwizzle == 2 || midSwizzle == 3) && (sizeK == BLOCK_K_HALF || sizeK == BLOCK_K_FULL))
struct SwizzleAtom {};

template<>
struct SwizzleAtom<cublasdx::arrangement::row_major, 2, BLOCK_K_FULL> {
    using swizzleAtom =  decltype(
    cute::composition(cute::Swizzle<3,2,3>{},
                cute::Layout<cute::Shape<cute::_8, cute::_8>,
                       cute::Stride<cute::_8, cute::_1>>{}));
};

template<>
struct SwizzleAtom<cublasdx::arrangement::col_major, 2, BLOCK_K_FULL> {
    using swizzleAtom =  decltype(
    composition(cute::Swizzle<3,2,3>{},
                cute::Layout<cute::Shape <cute::_8, cute::_8>,
                       cute::Stride< cute::_1,cute::_8>>{}));
};

template<>
struct SwizzleAtom<cublasdx::arrangement::row_major, 2, BLOCK_K_HALF> {
    using swizzleAtom =  decltype(
    cute::composition(cute::Swizzle<3,2,3>{},
                cute::Layout<cute::Shape < cute::_8,cute::_16>,
                       cute::Stride<cute::_16, cute::_1>>{}));
};

template<>
struct SwizzleAtom<cublasdx::arrangement::col_major, 2, BLOCK_K_HALF> {
    using swizzleAtom =  decltype(
    composition(cute::Swizzle<3,2,3>{},
                cute::Layout<cute::Shape <cute::_16, cute::_8>,
                       cute::Stride< cute::_1, cute::_16>>{}));
};

template<>
struct SwizzleAtom<cublasdx::arrangement::row_major, 3, BLOCK_K_FULL> {
    using swizzleAtom =  decltype(
    composition(cute::Swizzle<3,3,3>{},
                cute::Layout<cute::Shape < cute::_8,cute::_8>,
                       cute::Stride<cute::_8, cute::_1>>{}));
};

template<>
struct SwizzleAtom<cublasdx::arrangement::col_major, 3, BLOCK_K_FULL> {
    using swizzleAtom =  decltype(
    composition(cute::Swizzle<3,3,3>{},
                cute::Layout<cute::Shape <cute::_8, cute::_8>,
                       cute::Stride< cute::_1,cute::_8>>{}));
};

template<>
struct SwizzleAtom<cublasdx::arrangement::row_major, 3, BLOCK_K_HALF> {
    using swizzleAtom =  decltype(
    composition(cute::Swizzle<3,3,3>{},
                cute::Layout<cute::Shape < cute::_8,cute::_16>,
                       cute::Stride<cute::_16, cute::_1>>{}));
};

template<>
struct SwizzleAtom<cublasdx::arrangement::col_major, 3, BLOCK_K_HALF> {
    using swizzleAtom =  decltype(
    composition(cute::Swizzle<3,3,3>{},
                cute::Layout<cute::Shape <cute::_16, cute::_8>,
                       cute::Stride< cute::_1,cute::_16>>{}));
};

template<typename Element, unsigned int Arch>
using copyArch = cuda::std::conditional_t<sizeof(Element) >= 4 && Arch >= 800,
    cute::SM80_CP_ASYNC_CACHEALWAYS<Element>, cute::UniversalCopy<Element>>;

template<typename Element>
using sCopyLay = cuda::std::conditional_t<sizeof(Element) >= 4,
cute::AutoVectorizingCopyWithAssumedAlignment<8 * 16 / sizeof(Element)>, cute::SM75_U32x2_LDSM_N>;

template<
    typename ElementA,
    typename ElementB,
    unsigned int Arch,
    cublasdx::arrangement a = cublasdx::arrangement::row_major, // T
    cublasdx::arrangement b = cublasdx::arrangement::row_major  // N
>
struct CopyOp {
    static_assert((a == cublasdx::arrangement::row_major &&
        b == cublasdx::arrangement::row_major )||
        (a == cublasdx::arrangement::col_major &&
            b == cublasdx::arrangement::col_major));

    using copyAT = decltype(cute::make_tiled_copy(
        cute::Copy_Atom<copyArch<ElementA, Arch>, ElementA>{},
        cute::Layout<cute::Shape<cute::_16, cute::_8>,
            cute::Stride<cute::_8, cute::_1>>{},
        cute::Layout<cute::Shape<cute::_1, cute::_1>>{}));

    using copyBN = decltype(cute::make_tiled_copy(
        cute::Copy_Atom<copyArch<ElementB, Arch>, ElementB>{},
        cute::Layout<cute::Shape<cute::_16, cute::_8>,
            cute::Stride<cute::_8, cute::_1>>{},
        cute::Layout<cute::Shape<cute::_1, cute::_1>>{}));

    using copyAN = decltype(cute::make_tiled_copy(
        cute::Copy_Atom<copyArch<ElementA, Arch>, ElementA>{},
        cute::Layout<cute::Shape<cute::_16, cute::_8>>{},
        cute::Layout<cute::Shape<cute::_1, cute::_1>>{}));

    using copyBT = decltype(cute::make_tiled_copy(
        cute::Copy_Atom<copyArch<ElementB, Arch>, ElementB>{},
        cute::Layout<cute::Shape<cute::_16, cute::_8>>{},
        cute::Layout<cute::Shape<cute::_1, cute::_1>>{}));

    using copyA = cuda::std::conditional_t<(a == cublasdx::arrangement::row_major &&
        b == cublasdx::arrangement::row_major), copyAT, copyAN>;
    using copyB = cuda::std::conditional_t<(a == cublasdx::arrangement::row_major &&
        b == cublasdx::arrangement::row_major), copyBN, copyBT>;
};

enum class LayoutOptimization {
  UseSwizzle,
    UseVanilla
};

template<typename T>
requires (sizeof(T) == 2 || sizeof(T) == 4)
using MiddleSwizzle = cute::Int<sizeof(T) == 2 ? 3 : 2>;

template<class GEMM,
LayoutOptimization lOpt = LayoutOptimization::UseVanilla,
typename ElementA = toCT<typename GEMM::a_value_type>,
typename ElementB = toCT<typename GEMM::b_value_type>,
typename ElementC = toCT<typename GEMM::c_value_type>>
requires (cublasdx::is_complete_blas<GEMM>::value
&& cublasdx::sm_of<GEMM>::value >= MIN_ARCH
&& cublasdx::sm_of<GEMM>::value < 900)
struct CollectiveMMAConfig{
    using ldA = cuda::std::conditional_t<cublasdx::arrangement_of<GEMM>::a == cublasdx::row_major,
    cute::Int<cublasdx::size_of<GEMM>::k>, cute::Int<cublasdx::size_of<GEMM>::m>>; // A: (m,k)
    using ldB = cuda::std::conditional_t<cublasdx::arrangement_of<GEMM>::b == cublasdx::row_major,
    cute::Int<cublasdx::size_of<GEMM>::k>, cute::Int<cublasdx::size_of<GEMM>::n>>; // B: (n,k)
    using ldC = cuda::std::conditional_t<cublasdx::arrangement_of<GEMM>::c == cublasdx::row_major,
    cute::Int<cublasdx::size_of<GEMM>::n>, cute::Int<cublasdx::size_of<GEMM>::m>>; //C: (m,n)

    using copyAB = CopyOp<
        ElementA,
        ElementB,
        cublasdx::sm_of<GEMM>::value,
        cublasdx::arrangement_of<GEMM>::a,
        cublasdx::arrangement_of<GEMM>::b
    >;

    using gCopyA = typename copyAB::copyA;
    using gCopyB = typename copyAB::copyB;

    using sCopyA = cute::Copy_Atom<cuda::std::conditional_t<cublasdx::sm_of<GEMM>::value < 800,
    cute::AutoVectorizingCopyWithAssumedAlignment<8 * cublasdx::alignment_of<GEMM>::a>,
    sCopyLay<ElementA>>, ElementA>;
    using sCopyB = cute::Copy_Atom<cuda::std::conditional_t<cublasdx::sm_of<GEMM>::value < 800,
    cute::AutoVectorizingCopyWithAssumedAlignment<8 * cublasdx::alignment_of<GEMM>::b>,
    sCopyLay<ElementB>>, ElementB>;
    using sCopyC = cute::Copy_Atom<cute::AutoVectorizingCopyWithAssumedAlignment<8 * cublasdx::alignment_of<GEMM>::c>, ElementC>;

    using vSLayA = cute::Layout<cute::Shape<cute::Int<cublasdx::size_of<GEMM>::m>, cute::Int<cublasdx::size_of<GEMM>::k>>,
    cuda::std::conditional_t<cublasdx::arrangement_of<GEMM>::a == cublasdx::arrangement::col_major,
    cute::Stride<cute::_1, ldA>, cute::Stride<ldA, cute::_1>>>;
    using sLayA = cuda::std::conditional_t<lOpt == LayoutOptimization::UseSwizzle,
    typename SwizzleAtom<cublasdx::arrangement_of<GEMM>::a,
    MiddleSwizzle<ElementA>{}, cublasdx::size_of<GEMM>::k>::swizzleAtom, vSLayA>;

    using vSLayB = cute::Layout<cute::Shape<cute::Int<cublasdx::size_of<GEMM>::n>, cute::Int<cublasdx::size_of<GEMM>::k>>,
    cuda::std::conditional_t<cublasdx::arrangement_of<GEMM>::b == cublasdx::arrangement::col_major,
    cute::Stride<cute::_1, ldB>, cute::Stride<ldB, cute::_1>>>;
    using sLayB = cuda::std::conditional_t<lOpt == LayoutOptimization::UseSwizzle,
    typename SwizzleAtom<cublasdx::arrangement_of<GEMM>::b,
    MiddleSwizzle<ElementB>{}, cublasdx::size_of<GEMM>::k>::swizzleAtom, vSLayB>;

    using sLayC = cute::Layout<cute::Shape<cute::Int<cublasdx::size_of<GEMM>::m>, cute::Int<cublasdx::size_of<GEMM>::n>>,
    cuda::std::conditional_t<cublasdx::arrangement_of<GEMM>::c == cublasdx::arrangement::col_major,
    cute::Stride<cute::_1, ldC>, cute::Stride<ldC, cute::_1>>>;

    using mma_t = typename MMAConfig<cublasdx::sm_of<GEMM>::value, ElementC, ElementA,
    ElementB>::mma;
    using dispatch = cuda::std::conditional_t<cublasdx::sm_of<GEMM>::value < 800,
    cutlass::gemm::MainloopSm70TwoStageUnpredicated,
    cutlass::gemm::MainloopSm80CpAsyncUnpredicated<PIPELINE_STAGES>>;
};

template<
    unsigned int Arch,
    typename ElementA,
    typename ElementB,
    typename ElementC,
    typename ActivationOp = cute::identity
>
requires(cuda::std::is_same_v<ElementA, ElementB>)
struct BlockMM {
    static_assert(BLOCK_M == THREADS);
    static_assert(BLOCK_M == 128);
    static_assert(BLOCK_N == 64, "64 is a very good value for N, change it back!");
    using GEMM = decltype(cublasdx::Size<BLOCK_M, BLOCK_N, sizeof(ElementA) == 4 ? BLOCK_K_FULL : BLOCK_K_HALF>()
                          + cublasdx::Precision<toCDX<ElementA>, toCDX<ElementB>, toCDX<ElementC>>()
                          + cublasdx::Type<cublasdx::type::real>()
                          + cublasdx::Arrangement<cublasdx::row_major, cublasdx::row_major, cublasdx::row_major>()
                          + cublasdx::Function<cublasdx::function::MM>()
                          + cublasdx::SM<Arch>()
                          + cublasdx::Block()
                          + cublasdx::BlockDim<THREADS>());
    using MatrixAType = ElementA;
    using MatrixBType = ElementB;
    using MatrixCType = ElementC;
    using MatrixDType = ElementA;
    using BlockTiler = cute::Shape<cute::Int<cublasdx::size_of<GEMM>::m>,
                                    cute::Int<cublasdx::size_of<GEMM>::n>,
                                    cute::Int<cublasdx::size_of<GEMM>::k>>;
    using TilerOut = cute::Shape<cute::Int<cublasdx::size_of<GEMM>::m>, cute::Int<cublasdx::size_of<GEMM>::n>>;
    using Parameters = CollectiveMMAConfig<GEMM, LayoutOptimization::UseSwizzle>;
    using MMA = typename Parameters::mma_t;
    using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
        typename Parameters::dispatch,
        BlockTiler,
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
    using FusedEpilogue = FAA<ElementC, ActivationOp>;
};
#endif //TILING_CUH
