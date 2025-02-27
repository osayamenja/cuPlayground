/*#include <algorithm>
#include <random>
*/
#include <fmt/ranges.h>
#include <fmt/core.h>
#include "mma.cuh"
#include <cutlass/array.h>
#include <cutlass/numeric_conversion.h>

#include "util.cuh"

__global__ void theatre() {
    constexpr  auto sz = 16U;
    static_assert(sz % 16 == 0 && sz > 0);
    __shared__ __align__(16) float a [sz];
    for (uint i = 0; i < sz; ++i) {
        a[i] = static_cast<float>(i);
    }
    constexpr auto sCLay = make_layout(cute::Shape<cute::_4, cute::_4>{},
        cute::LayoutRight{});
    const auto sC = make_tensor(cute::make_smem_ptr(a), sCLay);
    print_tensor(sC);

}

int main() {
    /*constexpr std::array b{1.0f, 2.0f, 3.0f, 4.0f};
    std::array c{
        cute::tfloat32_t(4),
        cute::tfloat32_t(5),
        cute::tfloat32_t(6),
        cute::tfloat32_t(7)};
    constexpr cutlass::NumericArrayConverter<cute::tfloat32_t, float, 2> op{};
    const auto t = make_tensor(c.data(),
        make_layout(cute::make_shape(4, 1)));
    print_tensor(t);
    using ST = cutlass::AlignedArray<float, 2>;
    using RT = cutlass::AlignedArray<cute::tfloat32_t, 2>;
    static_assert(alignof(RT) == alignof(ST));
    CAST_TO(RT, c.data())[1] = *CAST_TO(RT, op(*CONST_CAST_TO(ST, &b[0])).data());
    print_tensor(t);*/
    testCollective();
}
