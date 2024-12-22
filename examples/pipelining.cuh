//
// Created by oja7 on 12/19/24.
//

#ifndef PIPELINING_CUH
#define PIPELINING_CUH

#include <cutlass/array.h>
#include "../util.cuh"

template<unsigned int processors = 400> requires (processors > 0)
__global__ __maxnreg__(128) void vanillaScheduler(const unsigned int* __restrict__ rQ,
    unsigned int* __restrict__ pDB,
    const __grid_constant__ unsigned int bound) {
    constexpr auto rounds = 1;
    #pragma unroll
    for (uint i = 0; i < rounds; ++i) {
        auto tasks = bound;
        auto rQt = 0U;
        auto scheduled = 0U;
        while (tasks) {
            const auto pid = rQ[rQt++ % processors];
            atomicExch(pDB + pid, ++scheduled);
            --tasks;
        }
    }
}

template<
    unsigned int processorCount = 400,
    unsigned int scratchSize = 64,
    unsigned int regSize = 32
>
requires (processorCount > 0 && scratchSize > 0 && regSize >= 32 &&
        scratchSize >= regSize && scratchSize % regSize == 0)
__global__ __maxnreg__(128) void fastScheduler(unsigned int* __restrict__ const& rQ,
    unsigned int* __restrict__ const& pDB,
    const __grid_constant__ unsigned int bound) {
    __shared__ __align__(16) unsigned int scratch[scratchSize];
    cutlass::AlignedArray<unsigned int, regSize> registerScratch;
    constexpr auto vectorLength = sizeof(uint4) / sizeof(unsigned int);
    static_assert(regSize % vectorLength == 0,
        "regSize must be a multiple of vectorLength");
    constexpr auto vectorSize = scratchSize / vectorLength;
    constexpr auto rounds = 1;
    #pragma unroll
    for (uint m = 0; m < rounds; ++m) {
        auto rQTail = 0U;
        auto tasks = bound;
        auto scheduled = 0U;
        while (tasks) {
            const auto trips = bound / scratchSize;
            if (trips) {
                #pragma unroll
                for (uint i = 0; i < vectorSize; i++) {
                    CAST_TO(uint4, scratch)[i] =
                        CAST_TO(uint4, rQ)[rQTail % processorCount];
                    rQTail += vectorLength;
                }
            }
            for (uint i = 0; i < trips; i++) {
                constexpr auto regTrips = scratchSize / regSize;
                #pragma unroll
                for (uint j = 0; j < regTrips; j++) {
                    #pragma unroll
                    for (uint k = 0; k < regSize; k++) {
                        registerScratch[k] = scratch[k + j * regSize];
                    }

                    // Eagerly prefetch next batch to smem
                    if (i + 1 < trips) {
                        constexpr auto rVs = regSize / vectorLength;
                        #pragma unroll
                        for (uint k = 0; k < rVs; k++) {
                            CAST_TO(uint4, scratch + j * regSize)[k] =
                                CAST_TO(uint4, rQ)[rQTail % processorCount];
                            rQTail += vectorLength;
                        }
                    }
                    #pragma unroll
                    for (uint k = 0; k < regSize; k++) {
                        const auto pid = registerScratch[k];
                        --tasks;
                        // Inform this process of a single task
                        atomicExch(pDB + pid, ++scheduled);
                    }
                }
            }
            // Handle residue
            if (const auto residue = bound - trips * scratchSize; residue) {
                const auto vR = residue / vectorLength;
                const auto rVr = residue - vR * vectorLength;
                for (uint i = 0; i < vR; i++) {
                    CAST_TO(uint4, scratch)[i] =
                        CAST_TO(uint4, rQ)[rQTail % processorCount];
                    rQTail += vectorLength;
                }
                // Residual shared trip
                for (uint i = 0; i < rVr; i++) {
                    // Stage in shared memory
                    scratch[i] = rQ[rQTail++ % processorCount];
                }

                const auto regTrips = residue / regSize;
                for (uint i = 0; i < regTrips; i++) {
                    #pragma unroll
                    for (uint j = 0; j < regSize; j++) {
                        registerScratch[j] = scratch[j + i * regSize];
                    }

                    #pragma unroll
                    for (uint j = 0; j < regSize; j++) {
                        const auto pid = registerScratch[j];
                        --tasks;
                        // Inform this process of a single task
                        atomicExch(pDB + pid, ++scheduled);
                    }
                }

                // Residual register trip
                const auto rRt = residue - regTrips * regSize;
                for (uint j = 0; j < rRt; ++j) {
                    registerScratch[j] = scratch[j + regTrips * regSize];
                }

                for (uint j = 0; j < rRt; j++) {
                    const auto pid = registerScratch[j];
                    --tasks;
                    // Inform this process of a single task
                    atomicExch(pDB + pid, ++scheduled);
                }
            }
        }
    }
}

enum class SchedulerType {
    vanilla,
    fast,
    asyncFast
};

template<SchedulerType s = SchedulerType::vanilla>
__host__ __forceinline__
void runScheduler(){
    uint* p;
    volatile uint bound = 4100;
    constexpr auto processors = 400;
    CHECK_ERROR_EXIT(cudaMallocAsync(&p, 2 * sizeof(unsigned int) * processors, cudaStreamPerThread));
    CHECK_ERROR_EXIT(cudaMemsetAsync(p, 0, 2 * sizeof(unsigned int) * processors, cudaStreamPerThread));
    /*for (uint i = 0; i < 128; ++i) {
        if constexpr (s == SchedulerType::vanilla) {
            vanillaScheduler<<<1, 1, 0, cudaStreamPerThread>>>(p, p + processors, bound);
        }
        else if constexpr (s == SchedulerType::fast) {
            fastScheduler<<<1, 1, 0, cudaStreamPerThread>>>(p, p + processors, bound);
        }
    }*/
    if constexpr (s == SchedulerType::vanilla) {
        vanillaScheduler<<<1, 1, 0, cudaStreamPerThread>>>(p, p + processors, bound);
    }
    else if constexpr (s == SchedulerType::fast) {
        fastScheduler<<<1, 1, 0, cudaStreamPerThread>>>(p, p + processors, bound);
    }
    CHECK_ERROR_EXIT(cudaFreeAsync(p, cudaStreamPerThread));
    CHECK_LAST();
}
#endif //PIPELINING_CUH
