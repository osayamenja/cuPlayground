#include <bitset>
#include <cuda/std/array>
#include <curand_kernel.h>
#include <fmt/ranges.h>
#include <fmt/core.h>
#include <nvshmem.h>
#include <cooperative_groups/memcpy_async.h>

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

template<unsigned int seed = 42, typename R = curandStatePhilox4_32_10_t>
requires(cuda::std::is_same_v<R, curandState> ||
    cuda::std::is_same_v<R, curandState_t> || cuda::std::is_same_v<R, curandStatePhilox4_32_10_t>)
__global__ void randK(void* __restrict__ rN) {
    auto* __restrict__ rN4 = CAST_TO(float4, rN);
    __shared__ R sharedStates[128];
    auto* __restrict__ cS = sharedStates + threadIdx.x;
    R rState{};
    curand_init(seed, threadIdx.x, 0, &rState);
    rN4[threadIdx.x] = curand_uniform4(&rState);
}

// uint64_t for signal type
__global__ void bench(uint64_t* __restrict__ p, const uint64_t q, const bool skip = true) {
    constexpr auto rounds = 64;
    double vC = 0.0f, aC = 0.0f; // vC -> volatile clocked, aC -> atomicClocked
    for (uint i = 0; i < q; ++i) {
        uint64_t start = 0, end = 0;
        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(start)::);
        __nv_atomic_compare_exchange_n(CAST_TO(uint16_t, p), CAST_TO(uint16_t, p), static_cast<uint16_t>(q),
            false, __NV_ATOMIC_RELAXED, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_SYSTEM);
        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(end)::);
        vC += static_cast<double>(end - start) / static_cast<double>(rounds);

        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(start)::);
        static_assert(sizeof(unsigned long long int) == sizeof(uint64_t));
        auto t = atomicExch_system(CAST_TO(unsigned long long int, p + 1), 0U);
        atomicCAS_system(CAST_TO(unsigned long long int, p + 1), 0U, t); // suggested alternative
        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(end)::);
        aC += static_cast<double>(end - start) / static_cast<double>(rounds);
    }

    if (!skip && !threadIdx.x) {
        printf("Block %u: vC is %f, aC is %f\n", blockIdx.x, vC, aC);
    }
}

__host__ __forceinline__
void hostBench() {
    uint64_t* p;
    volatile uint64_t q = 64U;
    CHECK_ERROR_EXIT(cudaMallocAsync(&p, sizeof(uint64_t) * 2, cudaStreamPerThread));
    /*for (uint i = 0; i < 128; ++i) {
        bench<<<1, 1, 0, cudaStreamPerThread>>>(p, q);
    }
    bench<<<1, 1, 0, cudaStreamPerThread>>>(p, q, false);*/
    bench<<<1, 1, 0, cudaStreamPerThread>>>(p, q);
    CHECK_ERROR_EXIT(cudaPeekAtLastError());
    CHECK_ERROR_EXIT(cudaDeviceSynchronize());
}

__host__ __forceinline__
void rH() {
    constexpr auto N = 128;
    float* dR;
    using Element = float;
    cuda::std::array<Element, N * 4> hR{};
    CHECK_ERROR_EXIT(cudaMallocAsync(&dR, hR.size() * sizeof(Element), cudaStreamPerThread));
    randK<<<1, N, 0, cudaStreamPerThread>>>(dR);
    CHECK_ERROR_EXIT(cudaMemcpyAsync(hR.data(), dR, sizeof(Element) * hR.size(),
        cudaMemcpyDeviceToHost, cudaStreamPerThread));
    CHECK_ERROR_EXIT(cudaPeekAtLastError());
    CHECK_ERROR_EXIT(cudaStreamSynchronize(cudaStreamPerThread));
    cuda::std::array<Element, 16> fHR{};
    std::memcpy(fHR.data(), hR.data(), sizeof(Element) * fHR.size());
    fmt::println("{}", fHR);
}

struct __align__(8) TQS {
    uint a;
    uint b;
};

__global__ void ExampleKernel(const unsigned long long int* __restrict__ d_data) {
    TQS t{0, 0};
    auto* __restrict__ tqsP = CAST_TO(unsigned long long int, &t);
    if (threadIdx.x % 32 == 0) {
        t.a = static_cast<uint>(d_data[threadIdx.x / 32]);
        t.b = static_cast<uint>(d_data[threadIdx.x / 32] + 1);
    }
    *tqsP = __shfl_sync(0xffffffff, *tqsP, 0);
}

__host__ __forceinline__
void doStore() {
    using Element = unsigned long long int;
    Element* p;
    constexpr auto n = 64;
    constexpr auto nt = 128;
    cuda::std::array<unsigned long long int, nt / 32> a{};
    volatile auto len = 4;
    for (uint i = 0; i < len; ++i) {
        a[i] = i;
    }
    CHECK_ERROR_EXIT(cudaMallocAsync(&p, sizeof(Element) * a.size(), cudaStreamPerThread));
    CHECK_ERROR_EXIT(cudaMemcpyAsync(p, a.data(), sizeof(Element) * a.size(), cudaMemcpyHostToDevice,
        cudaStreamPerThread));
    ExampleKernel<<<1, nt, 0, cudaStreamPerThread>>>(p);
    CHECK_ERROR_EXIT(cudaPeekAtLastError());
    CHECK_ERROR_EXIT(cudaStreamSynchronize(cudaStreamPerThread));
    //print_tensor(t);
}

template<
    unsigned int n,
    typename Element
>
__global__ void adK(const Element* __restrict__ p) {
    float cGcp = 0.0f, cTcp = 0.0f;
    __shared__ Element scratch[n];
    constexpr auto threads = 128;
    static_assert(n % threads == 0);
    for (uint i = 0; i < 128; ++i) {
        uint64_t start = 0, end = 0;
        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(start)::);
        memcpy_async(cooperative_groups::this_thread_block(), scratch, p, n * sizeof(Element));
        wait(cooperative_groups::this_thread_block());
        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(end)::);
        cGcp += static_cast<float>(end - start) / 128.0f;

        constexpr auto slice = n / threads;
        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(start)::);
        #pragma unroll
        for (uint j = 0; j < slice; ++j) {
            scratch[j * threads + threadIdx.x] = p[j * threads + threadIdx.x];
        }
        __syncthreads();
        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(end)::);
        cTcp += static_cast<float>(end - start) / 128.0f;
    }
    if (!threadIdx.x) {
        printf("cG is %f, cT is %f\n", cGcp, cTcp);
    }
}

uint16_t hsb = 1U;
#define THREADS 128U
#define WARP_SIZE 32U
#define SUBSCRIBERS (THREADS - WARP_SIZE)

__host__ __forceinline__
void ad() {
    using Element = float;
    constexpr auto n = 8 * 1024;
    Element* p;
    CHECK_ERROR_EXIT(cudaMallocAsync(&p, sizeof(Element) * n, cudaStreamPerThread));
    adK<n><<<1, 128, 0, cudaStreamPerThread>>>(p);
    CHECK_ERROR_EXIT(cudaPeekAtLastError());
    CHECK_ERROR_EXIT(cudaStreamSynchronize(cudaStreamPerThread));
}
int main() {
    testCollective();
    /*ad();
    constexpr cuda::std::array<float, 8> a {0, 1, 2, 3, 4,
        5, 6,7};
    const auto mA = make_tensor(a.data(), cute::Layout<cute::Shape<cute::_1, cute::_8>,
        cute::Stride<cute::_8, cute::_1>>{});
    const auto tC = idx2crd(4, cute::Shape<cute::_1, cute::_4>{},
        cute::Stride<cute::_4, cute::_1>{});
    const auto gA = local_tile(mA, cute::Shape<cute::_1, cute::_2>{},
        cute::get<1>(tC));
    print_tensor(gA);*/

}
