#include "mma.cuh"
#include <cuda/std/array>
#include <cute/tensor.hpp>
#include <cutlass/epilogue/thread/activation.h>

int main() {
    constexpr cutlass::epilogue::thread::ReLU<cute::half_t> f{};
    static_assert(cuda::std::is_invocable_r_v<cute::half_t, decltype(f), cute::half_t>);
    testP2PCollective();
}
