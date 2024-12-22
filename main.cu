#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cute/tensor.hpp>
#include <cuda/std/type_traits>

#include "mma.cuh"

__host__ __forceinline__
void tensorTheatre() {
    constexpr cuda::std::array<float, 16> arr {{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}};
    const auto mS = make_tensor(arr.data(),
                cute::Layout<cute::Shape<cute::_16, cute::_16>, cute::Stride<cute::_1, cute::_0>>{});
    print_tensor(mS);
}

int main() {
    using x = cuda::std::integral_constant<unsigned int, 4>;
    testCollective();
    //encore();
    //runScheduler<SchedulerType::vanilla>();
    //runScheduler<SchedulerType::fast>();
}