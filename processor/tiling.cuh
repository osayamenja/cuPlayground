//
// Created by osayamen on 10/24/24.
//

#ifndef TILING_CUH
#define TILING_CUH

#include <cublasdx.hpp>
#include <cuda/std/type_traits>
#include <cute/arch/copy.hpp>
#include <cute/arch/copy_sm80.hpp>

#define MIN_ARCH 700
#define THREADS 128

template<class GEMM>
requires (cublasdx::is_complete_blas<GEMM>::value
&& cublasdx::is_supported<GEMM, cublasdx::sm_of<GEMM>::value>::value
&& cublasdx::sm_of<GEMM>::value >= MIN_ARCH  && cublasdx::sm_of<GEMM>::value < 900)
struct GEMMParameters{
    using strideA = cuda::std::conditional_t<cublasdx::arrangement_of<GEMM>::a == cublasdx::row_major,
    cute::GenRowMajor, cute::GenColMajor>;
    using strideB = cuda::std::conditional_t<cublasdx::arrangement_of<GEMM>::b == cublasdx::row_major,
    cute::GenRowMajor, cute::GenColMajor>;
    using strideC = cuda::std::conditional_t<cublasdx::arrangement_of<GEMM>::c == cublasdx::row_major,
    cute::GenRowMajor, cute::GenColMajor>;

    // assert for NT
    using tLayX = cute::_32;
    using tLayY = cute::_8;

    static_assert(cublasdx::size_of<GEMM>::m % tLayX::value == 0);
    static_assert(cublasdx::size_of<GEMM>::n % tLayX::value == 0);
    static_assert(cublasdx::size_of<GEMM>::k % tLayY::value == 0);

    using vLayX = cute::Int<cublasdx::size_of<GEMM>::m / tLayX::value>;
    using vLayY = cute::Int<cublasdx::size_of<GEMM>::k / tLayY::value>;

    using SM70TiledCopyA = decltype(cute::make_tiled_copy(cute::Copy_Atom<cute::UniversalCopy<cute::uint128_t>,
                                    typename GEMM::a_value_type>{},
                                    cute::Layout<cute::Shape<tLayX, tLayY>>{},
                                    cute::Layout<cute::Shape<vLayX, vLayY>>{}));

    using SM70TiledCopyB = decltype(cute::make_tiled_copy(cute::Copy_Atom<cute::UniversalCopy<cute::uint128_t>,
                                    typename GEMM::b_value_type>{},
                                    cute::Layout<cute::Shape<tLayX, tLayY>>{},
                                    cute::Layout<cute::Shape<vLayX, vLayY>>{}));

    using SM80TiledCopyA = decltype(cute::make_tiled_copy(cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>,
                                    typename GEMM::a_value_type>{},
                                    cute::Layout<cute::Shape<tLayX, tLayY>>{},
                                    cute::Layout<cute::Shape<vLayX, vLayY>>{}));

    using SM80TiledCopyB = decltype(cute::make_tiled_copy(cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>,
                                    typename GEMM::b_value_type>{},
                                    cute::Layout<cute::Shape<tLayX, tLayY>>{},
                                    cute::Layout<cute::Shape<vLayX, vLayY>>{}));

    using gCopyA = cuda::std::conditional_t<(cublasdx::sm_of<GEMM>::value < 800), SM70TiledCopyA, SM80TiledCopyA>;
    using gCopyB = cuda::std::conditional_t<(cublasdx::sm_of<GEMM>::value < 800), SM70TiledCopyB, SM80TiledCopyB>;

    using config = cublasdx::detail::layout_database::optimal_config<128, cublasdx::sm_of<GEMM>::value,
    typename GEMM::a_value_type,
    cublasdx::arrangement_of<GEMM>::a == cublasdx::arrangement::col_major, cublasdx::alignment_of<GEMM>::a,
    typename GEMM::b_value_type,
    cublasdx::arrangement_of<GEMM>::b == cublasdx::arrangement::col_major, cublasdx::alignment_of<GEMM>::b,
    typename GEMM::c_value_type,
    cublasdx::arrangement_of<GEMM>::c == cublasdx::arrangement::col_major, cublasdx::alignment_of<GEMM>::c,
    cublasdx::size_of<GEMM>::m, cublasdx::size_of<GEMM>::n, cublasdx::size_of<GEMM>::k>;
};
#endif //TILING_CUH
