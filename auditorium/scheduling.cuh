//
// Created by oja7 on 12/26/24.
//

#ifndef SCHEDULING_CUH
#define SCHEDULING_CUH

#include <cute/numeric/math.hpp>
#include <cutlass/array.h>
#include "../util.cuh"

template<unsigned int processorCount, typename Registers>
requires(processorCount > 0 && isRegisterV<Registers>)
__device__ __forceinline__
void scheduleLoop(unsigned int& tasks,
    unsigned int& scheduled,
    const unsigned int& wSIdx,
    Registers& rtQTails,
    unsigned int* __restrict__ const& rQHead,
    unsigned int& rQTail,
    const unsigned int* __restrict__ const& rQ,
    const unsigned int& tQRow,
    const unsigned int& tQRl,
    unsigned int* __restrict__ const& pDB) {
    while (tasks) {
        const auto readyProcesses = atomicLoad<cuda::thread_scope_block>(rQHead) - rQTail;
        auto tasksToSchedule = cute::min(readyProcesses, tasks);
        tasks -= tasksToSchedule;
        scheduled += tasksToSchedule;
        while (tasksToSchedule) {
            // read pid from shared memory directly
            // batching to registers would be a pain to write :)
            const auto pid = rQ[rQTail++ % processorCount];
            --tasksToSchedule;
            ++rtQTails[wSIdx];
            // postfix index to differentiate from the sentinel signal; recipients deduct 1 before use
            atomicExch(pDB + pid, tQRow * tQRl + rtQTails[wSIdx]);
        }
    }
}

// all loops are unrolled
template<unsigned int processorCount, unsigned int producerCount, unsigned int wSetSize, typename Registers>
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
            scheduleLoop<processorCount>(tasks, scheduled, j, rtQTails, rQHead,
                rQTail, rQ, tQRow, tQRl, pDB);
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
        scheduleLoop<processorCount>(tasks, scheduled, j, rtQTails, rQHead,
                rQTail, rQ, tQRow, tQRl, pDB);
    }

    #pragma unroll
    for (uint j = 0; j < residue; ++j) {
        // persist
        tQTails[j + producerBatches * wSetSize] = rtQTails[j];
    }
}

// dynamic trip, used for external task queue whose heads are in global memory not shared.
template<unsigned int processorCount, unsigned int wSetSize, typename Registers>
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
    const unsigned int& taskMailboxes) {
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
            scheduleLoop<processorCount>(tasks, scheduled, j, rtQTails, rQHead,
                rQTail, rQ, tQRow, tQRl, pDB);
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
        scheduleLoop<processorCount>(tasks, scheduled, j, rtQTails, rQHead,
                rQTail, rQ, tQRow, tQRl, pDB);
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
    constexpr auto wSetSize = 64;
    cutlass::AlignedArray<unsigned int, wSetSize> rtQTails{};

    constexpr auto sB = 16;
    static_assert(sB % 4 == 0);
    const auto batches = gtQCL / sB;

    rtQTails.fill(0U);
    // memset
    #pragma unroll
    for (uint i = 0; i < producerCount; ++i) {
        tQTails[i] = 0U;
    }

    for (uint i = 0; i < batches; ++i) {
        #pragma unroll
        for (uint j = 0; j < sB; ++j) {
            tQTails[j] = 0U;
        }
    }
    for (uint i = batches * sB; i < gtQCL; ++i) {
        tQTails[i] = 0U;
    }

    while (scheduled < atomicLoad<cuda::thread_scope_block>(taskBound)) {
        // sweep all static wQHeads and schedule pending tasks
        staticSchedule<processorCount, producerCount, wSetSize>(scheduled, rtQTails, tQRl, tQTails,
            tQHeads, rQHead, rQTail, rQ, pDB);
        // Now do dynamic scheduling
        dynamicSchedule<processorCount, wSetSize>(scheduled, rtQTails, gtQRl, gtQTails,
            gtQHeads, rQHead, rQTail, rQ, pDB, gtQCL);
    }
}
#endif //SCHEDULING_CUH
