//
// Created by osayamen on 10/24/24.
//

#ifndef TILING_CUH
#define TILING_CUH

#include <cublasdx.hpp>
#include <cuda/std/type_traits>
#include <cute/arch/copy.hpp>
#include <cute/arch/copy_sm80.hpp>

// GEMM configuration constants
#define MIN_ARCH 700U
#define THREADS 128U
#define BLOCK_M 128U
#define BLOCK_N 64U
#define BLOCK_K_HALF 16U
#define BLOCK_K_FULL 8U
#define MAX_REGS (BLOCK_M * BLOCK_N) / THREADS

template<unsigned int Arch, typename TC, typename TA=TC, typename TB=TA>
struct MMAConfig {
    using mma = cute::TiledMMA<
                cute::MMA_Atom<cute::UniversalFMA<TC, TA, TB>>,
                cute::Layout<cute::Shape<cute::_16, cute::_8, cute::_1>>>;
};

template<>
struct MMAConfig<700, cute::half_t> {
    using mma = cute::TiledMMA<
      cute::MMA_Atom<cute::SM70_8x8x4_F16F16F16F16_TN>,
      cute::Layout<cute::Shape<cute::_4, cute::_4, cute::_1>>>;
};

template<>
struct MMAConfig<700, float, cute::half_t> {
    using mma = cute::TiledMMA<
      cute::MMA_Atom<cute::SM70_8x8x4_F32F16F16F32_TN>,
      cute::Layout<cute::Shape<cute::_4, cute::_4, cute::_1>>>;
};

template<>
struct MMAConfig<800, cute::half_t> {
    using mma = cute::TiledMMA<
      cute::MMA_Atom<cute::SM80_16x8x16_F16F16F16F16_TN>,
      cute::Layout<cute::Shape<cute::_4, cute::_1, cute::_1>>>;
};

template<>
struct MMAConfig<800, float, cute::half_t> {
    using mma = cute::TiledMMA<
      cute::MMA_Atom<cute::SM80_16x8x16_F32F16F16F32_TN>,
      cute::Layout<cute::Shape<cute::_4, cute::_1, cute::_1>>>;
};

template<>
struct MMAConfig<800, float, cute::bfloat16_t> {
    using mma = cute::TiledMMA<
      cute::MMA_Atom<cute::SM80_16x8x16_F32BF16BF16F32_TN>,
      cute::Layout<cute::Shape<cute::_4, cute::_1, cute::_1>>>;
};

template<>
struct MMAConfig<800, float, cute::tfloat32_t> {
    // TODO this may be incorrect
    using mma = cute::TiledMMA<
      cute::MMA_Atom<cute::SM80_16x8x8_F32TF32TF32F32_TN>,
      cute::Layout<cute::Shape<cute::_4, cute::_1, cute::_1>>>;
};

template <cublasdx::arrangement a, unsigned int sizeK>
struct SwizzleAtom;

template<>
struct SwizzleAtom<cublasdx::arrangement::row_major, BLOCK_K_HALF> {
    using swizzleAtom =  decltype(
    composition(cute::Swizzle<2,3,3>{},
                cute::Layout<cute::Shape < cute::_8, cute::_32>,
                       cute::Stride<cute::_32, cute::_1>>{}));
};

template<>
struct SwizzleAtom<cublasdx::arrangement::col_major, BLOCK_K_HALF> {
    // Weird combination but such is life
    using swizzleAtom =  decltype(
    composition(cute::Swizzle<3,3,3>{},
                cute::Layout<cute::Shape <cute::_64, cute::_8>,
                       cute::Stride< cute::_1,cute::_64>>{}));
};

template<>
struct SwizzleAtom<cublasdx::arrangement::row_major, BLOCK_K_FULL> {
    using swizzleAtom =  decltype(
    composition(cute::Swizzle<3,2,3>{},
                cute::Layout<cute::Shape < cute::_8,cute::_32>,
                       cute::Stride<cute::_32, cute::_1>>{}));
};

template<>
struct SwizzleAtom<cublasdx::arrangement::col_major, BLOCK_K_FULL> {
    // Weird combination but such is life
    using swizzleAtom =  decltype(
    composition(cute::Swizzle<3,2,3>{},
                cute::Layout<cute::Shape <cute::_32, cute::_8>,
                       cute::Stride< cute::_1,cute::_32>>{}));
};


enum class LayoutOptimization {
  UseSwizzle,
    UseVanilla
};

template<class GEMM, LayoutOptimization lOpt = LayoutOptimization::UseVanilla>
requires (cublasdx::is_complete_blas<GEMM>::value
&& cublasdx::is_supported<GEMM, cublasdx::sm_of<GEMM>::value>::value
&& cublasdx::sm_of<GEMM>::value >= MIN_ARCH
&& cublasdx::sm_of<GEMM>::value < 900)
struct CollectiveMMAConfig{
    using strideA = cuda::std::conditional_t<cublasdx::arrangement_of<GEMM>::a == cublasdx::row_major,
    cute::GenRowMajor, cute::GenColMajor>;
    using strideB = cuda::std::conditional_t<cublasdx::arrangement_of<GEMM>::b == cublasdx::row_major,
    cute::GenRowMajor, cute::GenColMajor>;
    using strideC = cuda::std::conditional_t<cublasdx::arrangement_of<GEMM>::c == cublasdx::row_major,
    cute::GenRowMajor, cute::GenColMajor>;

    // assert for NT
    using tLayX = cute::_16;
    using tLayY = cute::_8;

    static_assert(cublasdx::size_of<GEMM>::m % tLayX::value == 0 && cublasdx::size_of<GEMM>::m == BLOCK_M);
    static_assert(cublasdx::size_of<GEMM>::n % tLayX::value == 0 && cublasdx::size_of<GEMM>::n == BLOCK_N);
    static_assert(cublasdx::size_of<GEMM>::k % tLayY::value == 0);
    static_assert(sizeof(GEMM::a_value_type) <= 2 && sizeof(GEMM::b_value_type) <= 2 ?
        cublasdx::size_of<GEMM>::k == BLOCK_K_HALF : cublasdx::size_of<GEMM>::k == BLOCK_K_FULL);

    using vLayX = cute::Int<sizeof(cute::uint128_t) / sizeof(GEMM::a_value_type)>;
    using vLayY = cute::Int<cublasdx::size_of<GEMM>::k / tLayY::value>;

    using SM70TiledCopyA = decltype(cute::make_tiled_copy(cute::Copy_Atom<cute::UniversalCopy<cute::uint128_t>,
                                    typename GEMM::a_value_type>{},
                                    cute::Layout<cute::Shape<tLayX, tLayY>>{},
                                    cute::Layout<cute::Shape<vLayX, vLayY>>{}));

    using SM70TiledCopyB = decltype(cute::make_tiled_copy(cute::Copy_Atom<cute::UniversalCopy<cute::uint128_t>,
                                    typename GEMM::b_value_type>{},
                                    cute::Layout<cute::Shape<tLayX, tLayY>>{},
                                    cute::Layout<cute::Shape<vLayX, vLayY>>{}));

    using SM80CopyAtomA = cuda::std::conditional_t<sizeof(typename GEMM::a_value_type) >=4,
    cute::SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, cute::UniversalCopy<cute::uint128_t>>;

    using SM80TiledCopyA = decltype(cute::make_tiled_copy(cute::Copy_Atom<SM80CopyAtomA,
                                    typename GEMM::a_value_type>{},
                                    cute::Layout<cute::Shape<tLayX, tLayY>>{},
                                    cute::Layout<cute::Shape<vLayX, vLayY>>{}));

    using SM80CopyAtomB = cuda::std::conditional_t<sizeof(typename GEMM::b_value_type) >=4,
    cute::SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, cute::UniversalCopy<cute::uint128_t>>;

    using SM80TiledCopyB = decltype(cute::make_tiled_copy(cute::Copy_Atom<SM80CopyAtomB,
                                    typename GEMM::b_value_type>{},
                                    cute::Layout<cute::Shape<tLayX, tLayY>>{},
                                    cute::Layout<cute::Shape<vLayX, vLayY>>{}));

    using gCopyA = cuda::std::conditional_t<(cublasdx::sm_of<GEMM>::value < 800), SM70TiledCopyA, SM80TiledCopyA>;
    using gCopyB = cuda::std::conditional_t<(cublasdx::sm_of<GEMM>::value < 800), SM70TiledCopyB, SM80TiledCopyB>;

    using sCopyA = cute::Copy_Atom<cute::SM75_U32x4_LDSM_N, typename GEMM::a_value_type>;
    using sCopyB = cute::Copy_Atom<cute::SM75_U32x4_LDSM_N, typename GEMM::b_value_type>;
    using sCopyC = cute::Copy_Atom<cute::AutoVectorizingCopyWithAssumedAlignment<>, typename GEMM::c_value_type>;

    using vSLayA = cute::Layout<cute::Shape<cute::Int<cublasdx::size_of<GEMM>::m>, cute::Int<cublasdx::size_of<GEMM>::k>>,
    cuda::std::conditional_t<cublasdx::arrangement_of<GEMM>::a == cublasdx::arrangement::col_major,
    cute::GenColMajor, cute::GenRowMajor>>;
    using sLayA = cuda::std::conditional_t<lOpt == LayoutOptimization::UseSwizzle,
    SwizzleAtom<cublasdx::arrangement_of<GEMM>::a, cublasdx::size_of<GEMM>::k>, vSLayA>;

    using vSLayB = cute::Layout<cute::Shape<cute::Int<cublasdx::size_of<GEMM>::n>, cute::Int<cublasdx::size_of<GEMM>::k>>,
    cuda::std::conditional_t<cublasdx::arrangement_of<GEMM>::b == cublasdx::arrangement::col_major,
    cute::GenColMajor, cute::GenRowMajor>>;
    using sLayB = cuda::std::conditional_t<lOpt == LayoutOptimization::UseSwizzle,
    SwizzleAtom<cublasdx::arrangement_of<GEMM>::a, cublasdx::size_of<GEMM>::k>, vSLayA>;

    using sLayC = cute::Layout<cute::Shape<cute::Int<cublasdx::size_of<GEMM>::m>, cute::Int<cublasdx::size_of<GEMM>::n>>,
    cuda::std::conditional_t<cublasdx::arrangement_of<GEMM>::c == cublasdx::arrangement::col_major,
    cute::GenColMajor, cute::GenRowMajor>>;

    using mma_t = MMAConfig<cublasdx::sm_of<GEMM>::value, typename GEMM::c_value_type, typename GEMM::a_value_type,
    typename GEMM::b_value_type>;
};

#endif //TILING_CUH
