#include <iostream>

#include <cuda/std/array>
#include <fmt/ranges.h>
#include <nvshmemx.h>
#include <nvshmem.h>
#include <host/nvshmemx_api.h> // Makes CLion happy

#include <cuda/experimental/device.cuh>
#include "util.cuh"

template<unsigned int x = 2>
struct Task {
    // TODO use cuda std array
    cuda::std::array<uint*, x> f{};

    __device__ __forceinline__
    Task() = default;
    __device__ __forceinline__
    explicit Task(const cuda::std::array<uint*, x>& _f) {
        #pragma unroll
        for (uint i = 0; i < x; ++i) {
            f[i] = _f[i];
        }
    }
};

__global__ void atf(uint* a) {
    a[2] = 45;
    const cuda::std::array<uint*, 2> arr {a, a + 4};
    auto t = Task(arr);
    assert(t[0][2] == a[2]);
}
int main() {
    uint* p;
    CHECK_ERROR_EXIT(cudaMalloc(&p, 8 * sizeof(unsigned int)));
    atf<<<1,1>>>(p);
    CHECK_LAST();
}