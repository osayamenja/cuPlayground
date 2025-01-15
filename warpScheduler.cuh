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
    uint idx;
};

__device__
enum ReadySignal : unsigned int {
    observed,
    ready
};

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
    cutlass::AlignedArray<TQState, 64> tqState{};
    cutlass::AlignedArray<uint8_t, sQsL> sQState{};

    tqState.fill(TQState{0U,0U});

    constexpr auto subscribers = THREADS - wS;
    static_assert(subscribers % wS == 0);
    constexpr auto sL = subscribers / wS;
    constexpr auto tWs = decltype(tqState)::kElements - sL; // tasks per dynamic trip
    constexpr auto tf = tWs * wS;
    const uint dT = tf >= gtQCL ? 0U : (gtQCL - tf) /  (decltype(tqState)::kElements * wS);
    // cub stuff
    using WarpScan = cub::WarpScan<uint>;
    using WarpScanShort = cub::WarpScan<uint16_t>;
    auto* wSt = CAST_TO(WarpScan::TempStorage, workspace);
    auto* wSSt = CAST_TO(WarpScanShort::TempStorage, workspace + sizeof(WarpScan::TempStorage));
    unsigned int gRQIdx = 0U;
    auto taskTally = 0U;
    uint16_t processorTally = processors; // initially all processors are available
    while (scheduled < atomicLoad<cuda::thread_scope_block>(taskBound)) {
        // statically sweep tQ for tasks
        #pragma unroll
        for (uint i = 0; i < sL; ++i) {
            const auto tasks = atomicLoad<cuda::thread_scope_block>(tQHeads + i * wS + threadIdx.x) -
                tqState[i].tQTail;
            tqState[i].tasks = tasks;
            taskTally += tasks;
            if (tasks) {

            }
        }

        uint lTt = 0U; // local task tally
        #pragma unroll
        for (uint j = 0; j < tWs; ++j) {
            const auto tasks = atomicLoad<cuda::thread_scope_block>(gtQHeads + j * wS + threadIdx.x) -
                tqState[j].tQTail;
            tqState[j].tasks = tasks;
            lTt += tasks;
        }

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
            }
            gRQIdx += processorTally;
            // Below ensures writes to rQ in shared memory are visible
            __syncwarp();
            // schedule tasks
            const auto tasks = cute::min(processorTally, taskTally);
            scheduled += tasks;
            processorTally -= tasks;
            taskTally -= tasks;
            // these will get scheduled now
            if (lRQIdx < tasks) {
                auto tasksToSchedule = cute::min(tasks - lRQIdx, lTt);
                lTt -= tasksToSchedule;
                #pragma unroll
                for (uint j = 0; j < decltype(tqState)::kElements; ++j) {
                    if (tqState[j].tasks && tasksToSchedule) {
                        const auto canSchedule = cute::min(tasksToSchedule, tqState[j].tasks);
                        tasksToSchedule -= canSchedule;
                        tqState[j].tasks -= canSchedule;
                        const auto taskIdx = (j < 3? j * wS : tQRl * subscribers + (j - 3) * wS) +
                            threadIdx.x + tqState[j].tQTail;
                        for (uint k = 0; k < canSchedule; ++k) {
                            // signal processor
                            atomicExch(pDB + rQ[gRQIdx + lRQIdx++ % processors], taskIdx + k);
                        }
                    }
                }
            }
        }
    }
}
#endif //WARPSCHEDULER_CUH
