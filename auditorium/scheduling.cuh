//
// Created by oja7 on 12/26/24.
//

#ifndef SCHEDULING_CUH
#define SCHEDULING_CUH

#include <cute/numeric/math.hpp>
#include <cutlass/array.h>
#include "../util.cuh"

template<unsigned int qSize, typename Registers, typename Element>
requires(isRegisterV<Registers> &&
    cuda::std::is_same_v<typename Registers::value_type, Element>)
__device__ __forceinline__
void batchReadQ(Element* __restrict__& q,
    unsigned int& qTail,
    Registers& registers,
    unsigned int const& n) {
    // assert(__isShared(q))
    constexpr auto wSetSize = Registers::kElements;
    static_assert(sizeof(uint4) % sizeof(Element) == 0);
    static_assert(wSetSize % (sizeof(uint4) / sizeof(Element)) == 0,
        "Vector access is not applicable");
    constexpr auto vLen = sizeof(uint4) / sizeof(Element);
    constexpr auto vWSize = wSetSize / vLen;
    const auto trips = n / wSetSize;

    if (n >= wSetSize) {
        #pragma unroll
        for (uint i = 0; i < vWSize; ++i) {
            CAST_TO(uint4, registers)[i] = CAST_TO(uint4, q)[qTail % qSize];
            qTail += vLen;
        }
    }
    else {
        for (uint i = 0; i < n; ++i) {
            registers[i] = q[qTail++ % qSize];
        }
    }
}
/// Making processorCount a compile-time constant is not a functional requirement but rather strictly
/// for optimizing the modulo operation, which is incredibly expensive.
/// Benchmarks show an order of magnitude performance difference for runtime vs. compile-time evaluation
template<unsigned int processorCount, unsigned int wQSetSize>
requires(processorCount > 0 && wQSetSize > 1 && cutlass::ispow2(wQSetSize))
__device__ __forceinline__
void start(unsigned int* __restrict__ const& taskBound,
    unsigned int* __restrict__ const& rQHead,
    unsigned int* __restrict__ const& rQ,
    unsigned int* __restrict__ const& rQTail,
    unsigned int* __restrict__ const& wQHead,
    unsigned int* __restrict__ const& wQTail,
    uint2* __restrict__ const& wQ,
    unsigned int* __restrict__ const& pDB) {
    constexpr auto wQBatchSize = 32;
    /*assert(__isShared(workspace) &&
        __isShared(rQHead) &&
        __isShared(wQHead) &&
        __isShared(taskBound));*/
    unsigned int scheduled = 0U;

    // register copy of queue tails
    unsigned int rQTailX = *rQTail;
    unsigned int wQTailX = *wQTail;
    cutlass::AlignedArray<uint2, wQBatchSize> wQStage{};

    constexpr auto vLen = sizeof(uint4) / sizeof(uint2);
    constexpr auto vWSize = wQBatchSize / vLen;
    while (scheduled < atomicLoad<cuda::thread_scope_block>(taskBound)) {
        // Let's schedule these
        const auto workItems = atomicLoad<cuda::thread_scope_block>(wQHead) - wQTailX;
        const auto trips = workItems / wQBatchSize;
        auto totalTasks = 0U;
        for (uint i = 0; i < trips; ++i) {
            #pragma unroll
            for (uint j = 0; j < vWSize; ++j) {
                CAST_TO(uint4, wQStage.data())[j] = CAST_TO(uint4, wQ)[wQTailX % wQSetSize];
                wQTailX += vLen;
            }

            #pragma unroll
            for (uint j = 0; j < wQBatchSize; ++j) {
                totalTasks += wQStage[j].y;
            }

            // frees up space for the wQ observer
            atomicAdd_block(wQTail, wQBatchSize);
            uint workItemIdx = 0, taskIdx = 0;
            while (totalTasks) {
                auto tasksToSchedule = cute::min(atomicLoad<cuda::thread_scope_block>(rQHead) - rQTailX,
                totalTasks);
                totalTasks -= tasksToSchedule;
                scheduled += tasksToSchedule;
                while (tasksToSchedule) {
                    auto [tQStartIdx, nTasks] = wQStage[workItemIdx];
                    for (; taskIdx < nTasks && tasksToSchedule; ++taskIdx) {
                        --tasksToSchedule;
                        // Read pids directly from shared memory
                        // It is possible to batch these to register memory,
                        // but the resultant code would be too complex
                        const auto pid = rQ[rQTailX++ % processorCount];
                        // + 1 to differentiate from the sentinel signal,
                        // as such recipients deduct 1 to get the actual index
                        atomicExch(pDB + pid, tQStartIdx + taskIdx + 1);
                        // frees up space for the rQ observer
                        atomicIncrement<cuda::thread_scope_block>(rQTail);
                    }
                    // move to the next work item
                    workItemIdx += taskIdx == nTasks ? 1 : 0;
                    taskIdx = taskIdx == nTasks ? 0 : taskIdx;
                }
            }
        }
        // residue
        const auto residue = workItems - trips * wQBatchSize;
        totalTasks = 0U;
        for (uint j = 0; j < residue; ++j) {
            wQStage[j] = wQ[wQTailX++ % wQSetSize];
            totalTasks += wQStage[j].y;
        }
        atomicAdd_block(wQTail, residue);

        uint workItemIdx = 0, taskIdx = 0;
        while (totalTasks) {
            auto tasksToSchedule = cute::min(atomicLoad<cuda::thread_scope_block>(rQHead) - rQTailX,
            totalTasks);
            while (tasksToSchedule) {
                auto [tQStartIdx, nTasks] = wQStage[workItemIdx];
                for (; taskIdx < nTasks && tasksToSchedule; ++taskIdx) {
                    ++scheduled;
                    --tasksToSchedule;
                    const auto pid = rQ[rQTailX++ % processorCount];
                    atomicExch(pDB + pid, tQStartIdx + taskIdx + 1);
                    atomicIncrement(rQTail);
                }
                workItemIdx += taskIdx == nTasks ? 1 : 0;
                taskIdx = taskIdx == nTasks ? 0 : taskIdx;
            }
        }
    }
}
#endif //SCHEDULING_CUH
