#include <array>

#include <cub/cub.cuh>
#include <cutlass/array.h>

#include "util.cuh"
#include "fmt/args.h"

#define TIMING 1
__global__ void blockReduction(const float init, float* __restrict__ p, const bool skip = true) {
    constexpr auto bM = 128U;
    constexpr auto bN = 64U;
    using BlockReduce = cub::BlockReduce<float, bM>;
    __shared__ __align__(16) cuda::std::byte workspace[bN * sizeof(BlockReduce::TempStorage)];
    auto* wT = CAST_TO(BlockReduce::TempStorage, workspace);
    cutlass::AlignedArray<float, bN> accumulator{};

    #pragma unroll
    for (uint i = 0; i < bN; ++i) {
        accumulator[i] = static_cast<float>(threadIdx.x) + init;
    }
#if TIMING
    float clocked = 0.0f;
    constexpr auto trials = 1U;
    for (uint j = 0; j < trials; ++j) {
        uint64_t start, end;
        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(start)::);
#endif
        constexpr auto wS = 32U;
        float cache[bN / wS];
        // Reduce down the column, completes in about 9.5 𝜇s
        #pragma unroll
        for (uint i = 0; i < bN; ++i) {
            auto interim = BlockReduce(wT[i]).Sum(accumulator[i]);
            // thread0 has the aggregate, which it broadcasts to all threads in its warp
            interim = __shfl_sync(0xffffffff, interim , 0);
            // Each thread owns bN / warpSize elements in striped arrangement.
            // We duplicate this value layout across all warps in the block, but only use the first warp's values.
            cache[i / wS] = threadIdx.x % wS == i % wS? interim : cache[i / wS];
        }
        if (threadIdx.x < wS) {
            // Only the first warp aggregates atomically, as other warps have garbage values
            #pragma unroll
            for (uint i = 0; i < bN / wS; ++i) {
                atomicAdd(p + (threadIdx.x + i * wS),  cache[i]);
            }
        }
#if TIMING
        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(end)::);
        clocked += static_cast<float>(end - start) / static_cast<float>(trials);
    }
    if (!threadIdx.x && !skip) {
        printf("Block Reduction takes %.2fns and p[63] is %f\n", clocked, p[63]);
    }
#endif
}

__host__ __forceinline__
void hostBR() {
    volatile const float init = 1.0f;
    float* p;
    constexpr auto bN = 64U;
    const auto play = cudaStreamPerThread;
    CHECK_ERROR_EXIT(cudaMallocAsync(&p, sizeof(float) * bN, play));
    CHECK_ERROR_EXIT(cudaMemsetAsync(p, 0, sizeof(float) * bN, play));
#if TIMING
    for (uint i = 0; i < 128; ++i) {
        blockReduction<<<1, 128, 0, play>>>(init, p);
    }
#endif
    blockReduction<<<1, 128, 0, play>>>(init, p, false);
    CHECK_ERROR_EXIT(cudaFreeAsync(p, play));
    CHECK_LAST();
}

int main() {
    hostBR();
}