#include <cuda/std/cstddef>
#include "auditorium/combine.cuh"

using SignalStruct = cuda::std::pair<unsigned int, ushort2>;
__device__ __forceinline__
// buffer is an 8-byte array, which we split into
// 4-byte integer denoting batch index
// 2 bytes denoting blockM dimension
// 2 bytes identifying the communication stage
void encodeSignal(cuda::std::byte* __restrict__ const& buffer, const SignalStruct& s) {
    // assert(__isShared(buffer));
    *CAST_TO(SignalStruct, buffer) = s;
}

__device__ __forceinline__
auto decodeSignal(cuda::std::byte* __restrict__ const& buffer) {
    return *CAST_TO(SignalStruct, buffer);
}

__global__ void packetSignal() {
    extern __shared__ cuda::std::byte packetScratch[];
    constexpr auto s = SignalStruct{2, ushort2{128, 1}};
    encodeSignal(packetScratch, s);
    printf("Encode(bI = 2, bM = 128, s = 1) -> %lu\n", *CAST_TO(uint64_t, packetScratch));
    const auto signal = decodeSignal(packetScratch);
    printf("Decode -> bI: %u, bM: %u, s : %u", signal.first, signal.second.x, signal.second.y);
}

template<typename R>
requires isRegisterV<R>
__device__ __forceinline__
void encore(R &r) {
    static_assert(R::kElements == 32);
    r[0] = uint2{45, 89};
}

__global__ void theatre() {
    cutlass::AlignedArray<uint2, 32> a{};
    printf("Before, a[0].x: %u\n", a[0].x);
    encore(a);
    printf("After, a[0].x: %u", a[0].x);
}

int main() {
    //packetSignal<<<1, 1, 8, cudaStreamPerThread>>>();
    theatre<<<1,1>>>();
    CHECK_LAST();
    //hostCombine();
}