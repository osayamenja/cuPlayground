#include <cuda/std/cstddef>
#include <cute/int_tuple.hpp>
#include "util.cuh"
#include "auditorium/scheduling.cuh"

__global__ __maxnreg__(128) void theatre(unsigned int* __restrict__ p, const bool skip = true) {
    __shared__ __align__(16) cuda::std::byte scratch [3000];
    constexpr auto processorCount = 4 * 108U;
    constexpr auto producerCount = 126;

    constexpr auto M = 8192U;
    constexpr auto N = 4096U;
    constexpr auto Nx = 4096U;
    constexpr auto bM = 128U;
    constexpr auto bN = 64U;

    constexpr auto fTB = M / bM * (Nx / bN);
    constexpr auto sTB = M / bM * (N / bN);
    constexpr auto tQRl = cute::ceil_div(fTB, producerCount);
    constexpr auto gtQCl = M / bM;
    constexpr auto gtQRl = N / bN;

    auto* __restrict__ taskBound = CAST_TO(unsigned int, scratch);
    *taskBound = fTB + sTB;
    auto* __restrict__ tQHeads = taskBound + 1;
    auto* __restrict__ tQTails = tQHeads + producerCount;
    #pragma unroll
    for (uint i = threadIdx.x; i < producerCount; i += blockDim.x) {
        tQHeads[i] = fTB / producerCount;
        constexpr auto residue = fTB - fTB / producerCount * producerCount;
        tQHeads[i] += static_cast<unsigned int>(i < residue);
    }
    #pragma unroll
    for (uint i = threadIdx.x; i < gtQCl; i += blockDim.x) {
        p[i] = gtQRl;
    }

    auto* __restrict__ gTQTails = tQTails + producerCount;
    auto* __restrict__ rQHead = gTQTails + gtQCl;
    *rQHead = fTB + sTB; // this is actually cheating
    const auto* __restrict__ rQ = rQHead + 1;
    #pragma unroll
    for (uint i = threadIdx.x; i < processorCount; i += blockDim.x) {
        // done this way to preserve the const attribute of rQ
        (rQHead + 1)[i] = i;
    }

    __syncthreads();
    if (!threadIdx.x) {
        uint64_t start, end;
        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(start)::);
        schedulerStart<processorCount, producerCount>(tQRl, gtQCl, gtQRl,
                tQHeads, p, gTQTails, tQTails, taskBound, rQHead, rQ, p + gtQCl);
        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(end)::);
        if (!skip) {
            printf("Time taken is : %fus\n", static_cast<float>(end - start) / 1000.0f);
        }
    }
}


int main() {
    constexpr auto M = 8192U;
    constexpr auto bM = 128U;

    constexpr auto processorCount = 4 * 108U;
    constexpr auto gtQCl = M / bM;

    unsigned int* p;
    CHECK_ERROR_EXIT(cudaMalloc(&p, (processorCount + gtQCl) * sizeof(unsigned int)));
    #pragma unroll
    for (uint i = 0; i < 1; ++i) {
        theatre<<<1, 64>>>(p);
        CHECK_LAST();
        printf("Weird...\n");
    }
    theatre<<<1, 64>>>(p, false);
    CHECK_ERROR_EXIT(cudaFree(p));
    CHECK_LAST();
    //hostCombine();
}