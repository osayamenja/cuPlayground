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
#define BLOCK_N 128
#define BLOCK_K 8

template<unsigned int Arch, typename TC, typename TA=TC, typename TB=TA>
struct MMAConfig {
    using mma = cute::TiledMMA<
                cute::MMA_Atom<cute::UniversalFMA<TC, TA, TB>>,
                cute::Layout<cute::Shape<cute::_16, cute::_8, cute::_1>>>;
};

template<>
struct MMAConfig<700, cute::half_t> {
    using mma = void;
};

template<>
struct MMAConfig<700, float, cute::half_t> {
    using mma = void;
};

template<>
struct MMAConfig<800, cute::half_t> {
    using mma = void;
};

template<>
struct MMAConfig<800, float, cute::half_t> {
    using mma = void;
};

template<>
struct MMAConfig<800, float, cute::bfloat16_t> {
    using mma = void;
};

template<>
struct MMAConfig<800, float, cute::tfloat32_t> {
    using mma = void;
};

template<>
struct MMAConfig<900, cute::half_t> {
    using mma = void;
};

template<>
struct MMAConfig<900, float, cute::half_t> {
    using mma = void;
};

template<>
struct MMAConfig<900, float, cute::bfloat16_t> {
    using mma = void;
};

template<>
struct MMAConfig<900, float, cute::tfloat32_t> {
    using mma = void;
};

template<>
struct MMAConfig<900, float, cute::float_e4m3_t> {
    using mma = void;
};

template<>
struct MMAConfig<900, float, cute::float_e4m3_t, cute::float_e5m2_t> {
    using mma = void;
};

template<>
struct MMAConfig<900, float, cute::float_e5m2_t> {
    using mma = void;
};

template<class GEMM>
requires (cublasdx::is_complete_blas<GEMM>::value
&& cublasdx::is_supported<GEMM, cublasdx::sm_of<GEMM>::value>::value
&& cublasdx::sm_of<GEMM>::value >= MIN_ARCH  && cublasdx::sm_of<GEMM>::value < 900)
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

    static_assert(cublasdx::size_of<GEMM>::m % tLayX::value == 0);
    static_assert(cublasdx::size_of<GEMM>::n % tLayX::value == 0);
    static_assert(cublasdx::size_of<GEMM>::k % tLayY::value == 0);

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

    // TODO upgrade below layouts to Swizzle
    using sLayA = cute::Layout<cute::Shape<cute::Int<cublasdx::size_of<GEMM>::m>, cute::Int<cublasdx::size_of<GEMM>::k>>,
    cuda::std::conditional_t<cublasdx::arrangement_of<GEMM>::a == cublasdx::arrangement::col_major,
    cute::GenColMajor, cute::GenRowMajor>>;
    using sLayB = cute::Layout<cute::Shape<cute::Int<cublasdx::size_of<GEMM>::n>, cute::Int<cublasdx::size_of<GEMM>::k>>,
    cuda::std::conditional_t<cublasdx::arrangement_of<GEMM>::b == cublasdx::arrangement::col_major,
    cute::GenColMajor, cute::GenRowMajor>>;
    using sLayC = cute::Layout<cute::Shape<cute::Int<cublasdx::size_of<GEMM>::m>, cute::Int<cublasdx::size_of<GEMM>::n>>,
    cuda::std::conditional_t<cublasdx::arrangement_of<GEMM>::c == cublasdx::arrangement::col_major,
    cute::GenColMajor, cute::GenRowMajor>>;

    using config = cublasdx::detail::layout_database::optimal_config<THREADS, cublasdx::sm_of<GEMM>::value,
    typename GEMM::a_value_type, cublasdx::arrangement_of<GEMM>::a == cublasdx::arrangement::col_major, cublasdx::alignment_of<GEMM>::a,
    typename GEMM::b_value_type, cublasdx::arrangement_of<GEMM>::b == cublasdx::arrangement::col_major, cublasdx::alignment_of<GEMM>::b,
    typename GEMM::c_value_type, cublasdx::arrangement_of<GEMM>::c == cublasdx::arrangement::col_major, cublasdx::alignment_of<GEMM>::c,
    cublasdx::size_of<GEMM>::m, cublasdx::size_of<GEMM>::n, cublasdx::size_of<GEMM>::k>;

    using mma_t = MMAConfig<cublasdx::sm_of<GEMM>::value, typename GEMM::a_value_type, typename GEMM::b_value_type,
    typename GEMM::c_value_type>;
};

#endif //TILING_CUH
