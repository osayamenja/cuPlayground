//
// Created by oja7 on 1/14/25.
//

#ifndef WARPSCHEDULER_CUH
#define WARPSCHEDULER_CUH

#include <cub/cub.cuh>
#include <cuda/std/cstddef>
#include <cutlass/array.h>
#include "util.cuh"
#include "processor/tiling.cuh"

__device__
struct __align__(4) TQState {
    uint16_t tQTail;
    uint16_t tasks;
};

__device__
enum ReadySignal : unsigned int {
    observed,
    ready
};

template<
    unsigned int processors,
    typename WarpScan = cub::WarpScan<uint>,
    typename WarpScanShort = cub::WarpScan<uint16_t>,
    unsigned int wS = 32,
    unsigned int subscribers = THREADS - wS,
    unsigned int sL = subscribers / wS,
    typename SQState,
    typename TQState
>
requires (processors > 0 && wS == 32 &&
    cuda::std::is_same_v<WarpScan, cub::WarpScan<uint>> &&
    cuda::std::is_same_v<WarpScanShort, cub::WarpScan<uint16_t>> &&
    isRegisterV<SQState> && isRegisterV<TQState> && subscribers % wS == 0)
__device__ __forceinline__
void schedulerLoop(SQState& sQState, TQState& tqState,
    const unsigned int& tQRl,
    uint& lTt, uint& taskTally, uint& processorTally,
    uint& gRQIdx, bool& pTEpilog, uint& scheduled,
    typename WarpScan::TempStorage* __restrict__ const& wSt,
    typename WarpScanShort::TempStorage* __restrict__ const& wSSt,
    unsigned int* __restrict__ const& sQ,
    uint16_t* __restrict__ const& rQ,
    unsigned int* __restrict__ const& pDB,
    const bool isMedley = false) {
    uint lRQIdx;
    // things are about to get warped :)
    // Aggregate tally across the warp
    WarpScan(*wSt).InclusiveSum(lTt, lRQIdx, taskTally);
    lRQIdx -= lTt;
    while (taskTally) {
        // Find processors if we are not currently aware of any
        while (!processorTally) {
            // sweep sQ to identify ready processes
            uint16_t lPt = 0U; // local processor tally
            #pragma unroll
            for (uint j = threadIdx.x; j < processors; j += wS) {
                const auto readiness = atomicExch(sQ + j, observed) == ready;
                lPt += readiness;
                sQState[j / wS * wS + threadIdx.x] = readiness;
            }
            uint16_t startIdx;
            // Aggregate tally across the warp
            WarpScanShort(*wSSt).InclusiveSum(lPt, startIdx, processorTally);
            startIdx -= lPt;
            // write to rQ
            if (lPt) {
                #pragma unroll
                for (uint j = 0; j < decltype(sQState)::kElements; ++j) {
                    if (sQState[j]) {
                        // write ready process pid to rQ
                        rQ[(gRQIdx + startIdx++) % processors] = j * wS + threadIdx.x;
                    }
                }
            }
            pTEpilog = true;
        }
        if (pTEpilog) {
            pTEpilog = false;
            gRQIdx += processorTally;
            // Below ensures writes to rQ in shared memory are visible warp-wide
            __syncwarp();
        }
        // schedule tasks
        const auto tasks = cute::min(processorTally, taskTally);
        scheduled += tasks;
        processorTally -= tasks;
        taskTally -= tasks;
        // these will get scheduled now
        if (lRQIdx < tasks) {
            auto tasksToSchedule = cute::min(tasks - lRQIdx, lTt);
            lTt -= tasksToSchedule;
            if (!isMedley) {
                #pragma unroll
                for (uint j = 0; j < sL; ++j) {
                    if (tqState[j].tasks && tasksToSchedule) {
                        tqState[j].tasks = 0U;
                        const auto canSchedule = cute::min(tasksToSchedule, tqState[j].tasks);
                        tasksToSchedule -= canSchedule;
                        tqState[j].tasks -= canSchedule;
                        const auto taskIdx = j * wS + threadIdx.x + tqState[j].tQTail;
                        for (uint k = 0; k < canSchedule; ++k) {
                            // signal processor
                            atomicExch(pDB + rQ[gRQIdx + lRQIdx++ % processors], taskIdx + k);
                        }
                    }
                }
            }
            #pragma unroll
            for (uint j = sL; j < decltype(tqState)::kElements; ++j) {
                if (tqState[j].tasks && tasksToSchedule) {
                    tqState[j].tasks = 0U;
                    const auto canSchedule = cute::min(tasksToSchedule, tqState[j].tasks);
                    tasksToSchedule -= canSchedule;
                    tqState[j].tasks -= canSchedule;
                    const auto taskIdx = tQRl * subscribers + (j - 3) * wS + threadIdx.x + tqState[j].tQTail;
                    for (uint k = 0; k < canSchedule; ++k) {
                        // signal processor
                        atomicExch(pDB + rQ[gRQIdx + lRQIdx++ % processors], taskIdx + k);
                    }
                }
            }
        }
    }
}

template<unsigned int processors>
requires(processors > 0)
__device__ __forceinline__
void start(cuda::std::byte* workspace,
    const unsigned int& tQRl,
    const unsigned int& gtQCL,
    const unsigned int& gtQRl,
    unsigned int* __restrict__ const& tQHeads, // shared
    unsigned int* __restrict__ const& gtQHeads, // global
    unsigned int* __restrict__ const& taskBound, // shared
    uint16_t* __restrict__ const& rQ, // shared
    unsigned int* __restrict__ const& sQ, // global
    unsigned int* __restrict__ const& pDB) { //  global
    // initialize register buffers
    uint scheduled = 0U;
    constexpr auto wS = 32U;
    constexpr auto sQsL = cute::ceil_div(processors, wS);
    static_assert(sQsL <= 32);

    constexpr auto subscribers = THREADS - wS;
    static_assert(subscribers % wS == 0);
    constexpr auto sL = subscribers / wS;
    cutlass::AlignedArray<TQState, 16 + sL> tqState{};
    cutlass::AlignedArray<uint8_t, sQsL> sQState{};
    tqState.fill(TQState{0U,0U});
    constexpr auto dQL = decltype(tqState)::kElements - sL;
    const uint dT = gtQCL / (wS * dQL);

    // Other batches
    const uint dTx = gtQCL <= wS * dQL? 0U : (gtQCL - wS * dQL) / (wS * dQL);

    // cub stuff
    using WarpScan = cub::WarpScan<uint>;
    using WarpScanShort = cub::WarpScan<uint16_t>;
    auto* __restrict__ wSt = CAST_TO(WarpScan::TempStorage, workspace);
    auto* __restrict__ wSSt = CAST_TO(WarpScanShort::TempStorage, workspace + sizeof(WarpScan::TempStorage));
    uint gRQIdx = 0U;
    uint taskTally = 0U;
    uint16_t processorTally = processors; // initially, all processors are available, ensure that rQ has all pids
    bool pTEpilog = false;
    while (scheduled < atomicLoad<cuda::thread_scope_block>(taskBound)) {
        // statically sweep tQ for tasks
        uint lTt = 0U; // local task tally
        #pragma unroll
        for (uint i = 0; i < sL; ++i) {
            const auto tasks = atomicLoad<cuda::thread_scope_block>(tQHeads + i * wS + threadIdx.x) -
                tqState[i].tQTail;
            tqState[i].tasks = tasks;
            lTt += tasks;
        }

        // Abstract queues as a 3-D tensor (B, Q, T),
        // where B is the batch dimension or total queue / (Q * T);
        // Q is the number of queues a thread observes in one-pass;
        // and T is the number of threads in a warp
        if (dT > 0) {
            // special case, where i == 0
            #pragma unroll
            for (uint j = sL; j < decltype(tqState)::kElements; ++j) {
                const auto qIdx = wS * (j - sL) + threadIdx.x;
                const auto tasks = atomicLoad<cuda::thread_scope_block>(tQHeads + qIdx) -
                    tqState[j].tQTail;
                tqState[j].tasks = tasks;
                lTt += tasks;
            }
            // schedule observed tasks
            schedulerLoop<processors>(sQState, tqState, tQRl, lTt, taskTally,
                processorTally, gRQIdx, pTEpilog, scheduled, wSt, wSSt, sQ, rQ, pDB);

            for (uint i = 1; i < dT; ++i) {
                // Needed to enforce register storage
                #pragma unroll
                for (uint j = sL; j < decltype(tqState)::kElements; ++j) {
                    const auto qIdx = wS * (dQL * i + (j - sL)) + threadIdx.x;
                    const auto tasks = atomicLoad<cuda::thread_scope_block>(tQHeads + qIdx) -
                        tqState[j].tQTail;
                    tqState[j].tasks = tasks;
                    lTt += tasks;
                }
                // schedule observed tasks
                schedulerLoop<processors>(sQState, tqState, tQRl, lTt, taskTally,
                    processorTally, gRQIdx, pTEpilog, scheduled, wSt, wSSt, sQ, rQ, pDB, false);
            }
        }

        // residue
        #pragma unroll
        for (uint j = 0; j < dQL; ++j) {
            if (const auto qIdx = wS * (dQL * dT + j) + threadIdx.x; qIdx < gtQCL) {
                const auto tasks = atomicLoad<cuda::thread_scope_block>(gtQHeads + qIdx) -
                    tqState[j].tQTail;
                tqState[j].tasks = tasks;
                lTt += tasks;
            }
        }
        // schedule observed tasks
        schedulerLoop<processors>(sQState, tqState, tQRl, lTt, taskTally,
            processorTally, gRQIdx, pTEpilog, scheduled, wSt, wSSt, sQ, rQ, pDB, false);
    }
}
#endif //WARPSCHEDULER_CUH
