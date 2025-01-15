//
// Created by oja7 on 12/26/24.
//

#ifndef SCHEDULING_CUH
#define SCHEDULING_CUH

#include <cute/numeric/math.hpp>
#include <cutlass/array.h>
#include "../util.cuh"

template<unsigned int processorCount, unsigned int rQSetSize>
requires(processorCount > 0  && rQSetSize > 1 && cutlass::ispow2(rQSetSize))
__device__ __forceinline__
void scheduleLoop(unsigned int& tasks,
    unsigned int& scheduled,
    unsigned int& rtQTail,
    unsigned int* __restrict__ const& rQHead,
    unsigned int& rQTail,
    const unsigned int* __restrict__ const& rQ,
    const unsigned int& tQIdx,
    unsigned int* __restrict__ const& pDB) {
    while (tasks) {
        const auto readyProcesses = atomicLoad<cuda::thread_scope_block>(rQHead) - rQTail;
        const auto tasksToSchedule = cute::min(readyProcesses, tasks);
        tasks -= tasksToSchedule;
        scheduled += tasksToSchedule;
        if (tasksToSchedule) {
            // ensures reads to ready Q happen after signal reception
            __threadfence_block();
        }
        const auto tSb = tasksToSchedule / rQSetSize; // tasks to schedule batches
        for (uint i = 0; i < tSb; ++i) {
            #pragma unroll
            for (uint j = 0; j < rQSetSize; ++j) {
                const auto pid = rQ[rQTail++ % processorCount];
                ++rtQTail;
                atomicExch(pDB + pid, tQIdx + rtQTail);
            }
        }
        const auto residue = tasksToSchedule - tSb * rQSetSize;
        for (uint i = 0; i < residue; ++i) {
            const auto pid = rQ[rQTail++ % processorCount];
            ++rtQTail;
            // Fence is not needed prior to signal given task producers already issue one.
            atomicExch(pDB + pid, tQIdx + rtQTail);
        }
    }
}

// all loops are unrolled
template<unsigned int processorCount, unsigned int producerCount, unsigned int wSetSize, unsigned int rQSetSize,
typename Registers>
requires(processorCount > 0 && producerCount > 0 && producerCount < 128 && wSetSize > 1 && wSetSize % 4 == 0
    && isRegisterV<Registers> && Registers::kElements == wSetSize)
__device__ __forceinline__
void staticSchedule(unsigned int& scheduled,
    Registers& rtQTails,
    const unsigned int& tQRl,
    unsigned int* __restrict__ const& tQTails,
    unsigned int* __restrict__ const& tQHeads,
    unsigned int* __restrict__ const& rQHead,
    unsigned int& rQTail,
    const unsigned int* __restrict__ const& rQ,
    unsigned int* __restrict__ const& pDB) {
    constexpr auto producerBatches = producerCount / wSetSize;

    #pragma unroll
    for (uint i = 0; i < producerBatches; ++i) {
        // load wQ tails to registers
        #pragma unroll
        for (uint j = 0; j < wSetSize; ++j) {
            rtQTails[j] = tQTails[j + i * wSetSize];
        }
        #pragma unroll
        for (uint j = 0; j < wSetSize; ++j) {
            const auto tQRow = j + i * wSetSize;
            auto tasks = atomicLoad<cuda::thread_scope_block>(tQHeads + tQRow) - rtQTails[j];
            // Let's schedule these tasks
            scheduleLoop<processorCount, rQSetSize>(tasks, scheduled, rtQTails[j], rQHead,
                rQTail, rQ, tQRow * tQRl, pDB);
        }

        // persist wQ tails
        #pragma unroll
        for (uint j = 0; j < wSetSize; ++j) {
            tQTails[j + i * wSetSize] = rtQTails[j];
        }
    }
    constexpr auto residue = producerCount - producerBatches * wSetSize;
    #pragma unroll
    for (uint j = 0; j < residue; ++j) {
        // prefetch to registers
        rtQTails[j] = tQTails[j + producerBatches * wSetSize];
    }

    // complete static scheduling
    #pragma unroll
    for (uint j = 0; j < residue; ++j) {
        const auto tQRow = j + producerBatches * wSetSize;
        auto tasks = atomicLoad<cuda::thread_scope_block>(tQHeads + tQRow) - rtQTails[j];
        scheduleLoop<processorCount, rQSetSize>(tasks, scheduled, rtQTails[j], rQHead,
                rQTail, rQ, tQRow * tQRl, pDB);
    }

    #pragma unroll
    for (uint j = 0; j < residue; ++j) {
        // persist
        tQTails[j + producerBatches * wSetSize] = rtQTails[j];
    }
}

// dynamic trip, used for external task queue whose heads are in global memory not shared.
template<unsigned int processorCount, unsigned int wSetSize, unsigned int rQSetSize, typename Registers>
requires(processorCount > 0 && wSetSize > 1 && wSetSize % 4 == 0
    && isRegisterV<Registers> && Registers::kElements == wSetSize)
__device__ __forceinline__
void dynamicSchedule(unsigned int& scheduled,
    Registers& rtQTails,
    const unsigned int& tQRl,
    unsigned int* __restrict__ const& tQTails,
    unsigned int* __restrict__ const& tQHeads,
    unsigned int* __restrict__ const& rQHead,
    unsigned int& rQTail,
    const unsigned int* __restrict__ const& rQ,
    unsigned int* __restrict__ const& pDB,
    unsigned int const& taskMailboxes) {
    const auto mailboxBatches = taskMailboxes / wSetSize;

    for (uint i = 0; i < mailboxBatches; ++i) {
        // load wQ tails to registers
        #pragma unroll
        for (uint j = 0; j < wSetSize; ++j) {
            rtQTails[j] = tQTails[j + i * wSetSize];
        }
        #pragma unroll
        for (uint j = 0; j < wSetSize; ++j) {
            const auto tQRow = j + i * wSetSize;
            auto tasks = atomicLoad(tQHeads + tQRow) - rtQTails[j];
            // Let's schedule these tasks
            scheduleLoop<processorCount, rQSetSize>(tasks, scheduled, rtQTails[j], rQHead,
                rQTail, rQ, tQRow * tQRl, pDB);
        }

        // persist wQ tails
        #pragma unroll
        for (uint j = 0; j < wSetSize; ++j) {
            tQTails[j + i * wSetSize] = rtQTails[j];
        }
    }
    const auto residue = taskMailboxes - mailboxBatches * wSetSize;
    for (uint j = 0; j < residue; ++j) {
        // prefetch to registers
        rtQTails[j] = tQTails[j + mailboxBatches * wSetSize];
    }

    for (uint j = 0; j < residue; ++j) {
        const auto tQRow = j + mailboxBatches * wSetSize;
        auto tasks = atomicLoad(tQHeads + tQRow) - rtQTails[j];
        scheduleLoop<processorCount, rQSetSize>(tasks, scheduled, rtQTails[j], rQHead,
                rQTail, rQ, tQRow * tQRl, pDB);
    }

    for (uint j = 0; j < residue; ++j) {
        // persist
        tQTails[j + mailboxBatches * wSetSize] = rtQTails[j];
    }
}

template<unsigned int processorCount, unsigned int producerCount>
requires(processorCount > 0 && producerCount > 0 && producerCount < 128)
__device__ __forceinline__
void schedulerStart(const unsigned int& tQRl,
    const unsigned int& gtQCL,
    const unsigned int& gtQRl,
    unsigned int* __restrict__ const& tQHeads,
    unsigned int* __restrict__ const& gtQHeads,
    unsigned int* __restrict__ const& gtQTails,
    unsigned int* __restrict__ const& tQTails,
    unsigned int* __restrict__ const& taskBound,
    unsigned int* __restrict__ const& rQHead,
    const unsigned int* __restrict__ const& rQ,
    unsigned int* __restrict__ const& pDB) {
    // assert(__isShared(all arguments above) and alignment is 16)
    unsigned int scheduled = 0U;
    unsigned int rQTail = 0U;
    constexpr auto wSetSize = 16;
    cutlass::AlignedArray<unsigned int, wSetSize> rtQTails{};
    rtQTails.fill(0U);

    while (scheduled < atomicLoad<cuda::thread_scope_block>(taskBound)) {
        constexpr auto rQSetSize = 16;
        // sweep all static wQHeads and schedule pending tasks
        staticSchedule<processorCount, producerCount, wSetSize, rQSetSize>(scheduled, rtQTails, tQRl, tQTails,
            tQHeads, rQHead, rQTail, rQ, pDB);
        // Now do dynamic scheduling
        dynamicSchedule<processorCount, wSetSize, rQSetSize>(scheduled, rtQTails, gtQRl, gtQTails,
            gtQHeads, rQHead, rQTail, rQ, pDB, gtQCL);
    }
}

template<
    unsigned int processorCount,
    unsigned int producerCount,
    unsigned int M,
    unsigned int N,
    unsigned int Nx,
    unsigned int bM,
    unsigned int bN
>
__global__ __maxnreg__(128) void sK(unsigned int* __restrict__ p, const bool skip = true) {
    constexpr auto fTB = M / bM * (Nx / bN);
    constexpr auto sTB = M / bM * (N / bN);
    constexpr auto tQRl = cute::ceil_div(fTB, producerCount);
    constexpr auto gtQCl = M / bM;
    constexpr auto gtQRl = N / bN;

    constexpr auto sharedSize = (2 + 2 * producerCount + gtQCl + processorCount) * sizeof(unsigned int);
    __shared__ __align__(16) cuda::std::byte scratch [sharedSize];

    auto* __restrict__ taskBound = CAST_TO(unsigned int, scratch);
    *taskBound = fTB + sTB;
    auto* __restrict__ tQHeads = taskBound + 1;
    auto* __restrict__ tQTails = tQHeads + producerCount;
    #pragma unroll
    for (uint i = threadIdx.x; i < producerCount; i += blockDim.x) {
        tQHeads[i] = fTB / producerCount;
        constexpr auto residue = fTB - fTB / producerCount * producerCount;
        tQHeads[i] += static_cast<unsigned int>(i < residue);
        // clear state
        tQTails[i] = 0U;
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

    #pragma unroll
    for (uint i = threadIdx.x; i < gtQCl; i += blockDim.x) {
        // clear state
        gTQTails[i] = 0U;
    }

    __syncthreads();
    if (!threadIdx.x) {
        uint64_t start, end;
        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(start)::);
        schedulerStart<processorCount, producerCount>(tQRl, gtQCl, gtQRl,
                tQHeads, p, gTQTails, tQTails, taskBound, rQHead, rQ, p + gtQCl);
        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(end)::);
        if(!skip) {
            printf("Time taken is %fus\n", static_cast<float>(end - start) / 1000.0f);
        }
    }
}

__host__ __forceinline__
void hostSchedule() {
    constexpr auto M = 8192U;
    constexpr auto N = 4096U;
    constexpr auto Nx = 4096U;
    constexpr auto bM = 128U;
    constexpr auto bN = 64U;

    constexpr auto processorCount = 4 * 108U; // Ampere with <= 128 registers
    constexpr auto threads = 128;
    constexpr auto producerCount = threads - 2;
    constexpr auto gtQCl = M / bM;

    unsigned int* p;
    CHECK_ERROR_EXIT(cudaMallocAsync(&p, (processorCount + gtQCl) * sizeof(unsigned int), cudaStreamPerThread));
    for (uint i = 0; i < 256; ++i) {
        sK<processorCount, producerCount, M, N, Nx, bM, bN><<<1, 256, 0, cudaStreamPerThread>>>(p);
    }
    sK<processorCount, producerCount, M, N, Nx, bM, bN><<<1, 256, 0, cudaStreamPerThread>>>(p, false);
    CHECK_ERROR_EXIT(cudaFreeAsync(p, cudaStreamPerThread));
    CHECK_LAST();
}
#endif //SCHEDULING_CUH
