#include <iostream>

#include <cuda/std/cassert>
#include <cuda/std/array>
#include <fmt/ranges.h>
#include <nvshmemx.h>
#include <nvshmem.h>
#include <host/nvshmemx_api.h> // Makes CLion happy

#include <cuda/experimental/device.cuh>
#include "util.cuh"

struct Task {
    // TODO use cuda std array
    cuda::std::array<uint*, 2> f = {};

    __device__ __forceinline__
    Task() = default;

    __device__ __forceinline__
    explicit Task(const cuda::std::array<uint*, 2>& _f) : f(_f) {}
};

__global__ void theatre(uint* a) {
    __shared__ Task t;
    a[0] = 46;
    a[1] = 89;
    a[2] = 45;
    a[3] = 90;

    cuda::std::array<uint*, 2> arr {{a, a + 2}};
    t = Task(arr);
    static_assert(sizeof(Task) == sizeof(decltype(arr)::value_type) * arr.size());
    assert(t.f[0][1] == a[1]);
    t.f = arr;
    assert(t.f[0][3] == 90);
}
int main() {
    uint* p;
    CHECK_ERROR_EXIT(cudaMalloc(&p, 8 * sizeof(unsigned int)));
    theatre<<<1,1>>>(p);
    CHECK_LAST();
}