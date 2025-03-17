#include <bitset>

#include <cuda/std/array>
#include <curand_kernel.h>
#include <fmt/ranges.h>
#include <fmt/core.h>

#include "mma.cuh"
#include "util.cuh"

__host__ __forceinline__
void bitManip() {
    unsigned int x = 1U;
    std::cout << (x >> 0 & 1) << std::endl;
    std::cout << (x >> 31 & 1) << std::endl;
    x |= 1U << 31U;
    std::cout << std::bitset<32>(x) << std::endl;
    std::cout << (x >> 31 & 1) << std::endl;
}

template<unsigned int seed = 42, typename R>
requires(cuda::std::is_same_v<R, curandState> ||
    cuda::std::is_same_v<R, curandState_t> || cuda::std::is_same_v<R, curandStatePhilox4_32_10_t>)
__global__ void randK(R* __restrict__ states, void* __restrict__ rN) {
    auto* __restrict__ rN4 = CAST_TO(float4, rN);
    auto* __restrict__ cS = states + threadIdx.x;
    curand_init(seed, threadIdx.x, 0, cS);
    rN4[threadIdx.x] = curand_uniform4(cS);
}


int main() {
    constexpr auto N = 128;
    using R = curandStatePhilox4_32_10_t;
    R *dS;
    float* dR;
    using Element = float;
    cuda::std::array<Element, N * 4> hR{};
    CHECK_ERROR_EXIT(cudaMallocAsync(&dS, N * sizeof(R), cudaStreamPerThread));
    CHECK_ERROR_EXIT(cudaMallocAsync(&dR, hR.size() * sizeof(Element), cudaStreamPerThread));
    randK<<<1, N, 0, cudaStreamPerThread>>>(dS, dR);
    CHECK_ERROR_EXIT(cudaMemcpyAsync(hR.data(), dR, sizeof(Element) * hR.size(),
        cudaMemcpyDeviceToHost, cudaStreamPerThread));
    CHECK_ERROR_EXIT(cudaPeekAtLastError());
    CHECK_ERROR_EXIT(cudaStreamSynchronize(cudaStreamPerThread));
    cuda::std::array<Element, 16> fHR{};
    std::memcpy(fHR.data(), hR.data(), sizeof(Element) * fHR.size());
    fmt::println("{}", fHR);
    //bitManip();
    //testCollective();
}
