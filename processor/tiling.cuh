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
#define MIN_ARCH 700
#define THREADS 128
#define BLOCK_M 128
#define BLOCK_M_EXP 64
#define BLOCK_N 64
#define BLOCK_K_HALF 16
#define BLOCK_K_FULL 8
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
      cute::Layout<cute::Shape<cute::_2, cute::_2, cute::_1>>>;
};

template<>
struct MMAConfig<800, float, cute::half_t> {
    using mma = cute::TiledMMA<
      cute::MMA_Atom<cute::SM80_16x8x16_F32F16F16F32_TN>,
      cute::Layout<cute::Shape<cute::_2, cute::_2, cute::_1>>>;
};

template<>
struct MMAConfig<800, float, cute::bfloat16_t> {
    using mma = cute::TiledMMA<
      cute::MMA_Atom<cute::SM80_16x8x16_F32BF16BF16F32_TN>,
      cute::Layout<cute::Shape<cute::_2, cute::_2, cute::_1>>>;
};

template<>
struct MMAConfig<800, float, cute::tfloat32_t> {
    // TODO this may be incorrect
    using mma = cute::TiledMMA<
      cute::MMA_Atom<cute::SM80_16x8x8_F32TF32TF32F32_TN>,
      cute::Layout<cute::Shape<cute::_2, cute::_2, cute::_1>>>;
};

template <cublasdx::arrangement a, unsigned int sizeK>
struct SwizzleAtom;

template<>
struct SwizzleAtom<cublasdx::arrangement::row_major, BLOCK_K_HALF> {
    using swizzleAtom =  decltype(
    cute::composition(cute::Swizzle<3,3,3>{},
                cute::Layout<cute::Shape < cute::_8,cute::_64>,
                       cute::Stride<cute::_64, cute::_1>>{}));
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

template<unsigned int precision = 4,
        unsigned int bM = BLOCK_M,
        unsigned int bN = BLOCK_N,
        unsigned int bK = BLOCK_K_FULL,
        unsigned int threads = THREADS,
        cublasdx::arrangement a = cublasdx::arrangement::row_major,
        cublasdx::arrangement b = cublasdx::arrangement::col_major
>
struct ThreadCopyLayout {
    using tLayAX = cute::_16;
    using tLayAY = cute::_8;

    using tLayBX = cute::_16;
    using tLayBY = cute::_8;

    using vLayAX = cute::_8;
    using vLayAY = cute::_1;

    using vLayBX = cute::_4;
    using vLayBY = cute::_1;
};

template<>
struct ThreadCopyLayout<4, BLOCK_M_EXP> {
    using tLayAX = cute::_16;
    using tLayAY = cute::_8;

    using tLayBX = cute::_16;
    using tLayBY = cute::_8;

    using vLayAX = cute::_4;
    using vLayAY = cute::_1;

    using vLayBX = cute::_4;
    using vLayBY = cute::_1;
};

template<>
struct ThreadCopyLayout<2> {
    using tLayAX = cute::_16;
    using tLayAY = cute::_8;

    using tLayBX = cute::_16;
    using tLayBY = cute::_8;

    using vLayAX = cute::_8;
    using vLayAY = cute::_1;

    using vLayBX = cute::_8;
    using vLayBY = cute::_1;
};

template<>
struct ThreadCopyLayout<2, BLOCK_M_EXP> : ThreadCopyLayout<2>{};

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

    using tCopyLayout = ThreadCopyLayout<sizeof(typename GEMM::a_value_type),
        cublasdx::size_of<GEMM>::m,
        cublasdx::size_of<GEMM>::n,
        cublasdx::size_of<GEMM>::k,
        THREADS,
        cublasdx::arrangement_of<GEMM>::a,
        cublasdx::arrangement_of<GEMM>::b>;

    using SM70TiledCopyA = decltype(cute::make_tiled_copy(cute::Copy_Atom<cute::UniversalCopy<cute::uint128_t>,
                                    typename GEMM::a_value_type>{},
                                    cute::Layout<cute::Shape<typename tCopyLayout::tLayAX, typename tCopyLayout::tLayAY>>{},
                                    cute::Layout<cute::Shape<typename tCopyLayout::vLayAX, typename tCopyLayout::vLayAY>>{}));

    using SM70TiledCopyB = decltype(cute::make_tiled_copy(cute::Copy_Atom<cute::UniversalCopy<cute::uint128_t>,
                                    typename GEMM::b_value_type>{},
                                    cute::Layout<cute::Shape<typename tCopyLayout::tLayBX, typename tCopyLayout::tLayBY>>{},
                                    cute::Layout<cute::Shape<typename tCopyLayout::vLayBX, typename tCopyLayout::vLayBY>>{}));

    using SM80CopyAtomA = cuda::std::conditional_t<sizeof(typename GEMM::a_value_type) >=4,
    cute::SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, cute::UniversalCopy<cute::uint128_t>>;

    using SM80TiledCopyA = decltype(cute::make_tiled_copy(cute::Copy_Atom<SM80CopyAtomA,
                                    typename GEMM::a_value_type>{},
                                    cute::Layout<cute::Shape<typename tCopyLayout::tLayAX, typename tCopyLayout::tLayAY>>{},
                                    cute::Layout<cute::Shape<typename tCopyLayout::vLayAX, typename tCopyLayout::vLayAY>>{}));

    using SM80CopyAtomB = cuda::std::conditional_t<sizeof(typename GEMM::b_value_type) >=4,
    cute::SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, cute::UniversalCopy<cute::uint128_t>>;

    using SM80TiledCopyB = decltype(cute::make_tiled_copy(cute::Copy_Atom<SM80CopyAtomB,
                                    typename GEMM::b_value_type>{},
                                    cute::Layout<cute::Shape<typename tCopyLayout::tLayBX, typename tCopyLayout::tLayBY>>{},
                                    cute::Layout<cute::Shape<typename tCopyLayout::vLayBX, typename tCopyLayout::vLayBY>>{}));

    using gCopyA = cuda::std::conditional_t<(cublasdx::sm_of<GEMM>::value < 800), SM70TiledCopyA, SM80TiledCopyA>;
    using gCopyB = cuda::std::conditional_t<(cublasdx::sm_of<GEMM>::value < 800), SM70TiledCopyB, SM80TiledCopyB>;

    using sCopyA = cute::Copy_Atom<cute::SM75_U32x4_LDSM_N, typename GEMM::a_value_type>;
    using sCopyB = cute::Copy_Atom<cute::SM75_U32x4_LDSM_N, typename GEMM::b_value_type>;
    using sCopyC = cute::Copy_Atom<cute::AutoVectorizingCopyWithAssumedAlignment<>, typename GEMM::c_value_type>;

    using vSLayA = cute::Layout<cute::Shape<cute::Int<cublasdx::size_of<GEMM>::m>, cute::Int<cublasdx::size_of<GEMM>::k>>,
    cuda::std::conditional_t<cublasdx::arrangement_of<GEMM>::a == cublasdx::arrangement::col_major,
    cute::Stride<cute::_1, cute::Int<GEMM::lda>>, cute::Stride<cute::Int<GEMM::lda>, cute::_1>>>;
    using sLayA = cuda::std::conditional_t<lOpt == LayoutOptimization::UseSwizzle,
    SwizzleAtom<cublasdx::arrangement_of<GEMM>::a, cublasdx::size_of<GEMM>::k>, vSLayA>;

    using vSLayB = cute::Layout<cute::Shape<cute::Int<cublasdx::size_of<GEMM>::n>, cute::Int<cublasdx::size_of<GEMM>::k>>,
    cuda::std::conditional_t<cublasdx::arrangement_of<GEMM>::b == cublasdx::arrangement::col_major,
    cute::Stride<cute::_1, cute::Int<GEMM::ldb>>, cute::Stride<cute::Int<GEMM::ldb>, cute::_1>>>;
    using sLayB = cuda::std::conditional_t<lOpt == LayoutOptimization::UseSwizzle,
    SwizzleAtom<cublasdx::arrangement_of<GEMM>::b, cublasdx::size_of<GEMM>::k>, vSLayB>;

    using sLayC = cute::Layout<cute::Shape<cute::Int<cublasdx::size_of<GEMM>::m>, cute::Int<cublasdx::size_of<GEMM>::n>>,
    cuda::std::conditional_t<cublasdx::arrangement_of<GEMM>::c == cublasdx::arrangement::col_major,
    cute::Stride<cute::_1, cute::Int<GEMM::ldc>>, cute::Stride<cute::Int<GEMM::ldc>, cute::_1>>>;

    using mma_t = typename MMAConfig<cublasdx::sm_of<GEMM>::value, typename GEMM::c_value_type, typename GEMM::a_value_type,
    typename GEMM::b_value_type>::mma;
};

#endif //TILING_CUH
