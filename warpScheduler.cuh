//
// Created by oja7 on 1/14/25.
//

#ifndef WARPSCHEDULER_CUH
#define WARPSCHEDULER_CUH

#include <cub/cub.cuh>
#include <cuda/std/cstddef>
#include <cutlass/array.h>
#include "util.cuh"

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
    unsigned int wS = 32,
    unsigned int subscribers = 128 - wS,
    unsigned int sL = subscribers / wS,
    typename SQState,
    typename TQState
>
requires (processors > 0 && wS == 32 &&
    cuda::std::is_same_v<WarpScan, cub::WarpScan<uint>> &&
    isRegisterV<SQState> && isRegisterV<TQState> && subscribers % wS == 0)
__device__ __forceinline__
void schedulerLoop(SQState& sQState, TQState& tqState,
    const unsigned int& tQRl,
    uint& lTt, uint& taskTally, uint& processorTally,
    uint& gRQIdx, bool& pTEpilog, uint& scheduled,
    typename WarpScan::TempStorage* __restrict__ const& wSt,
    unsigned int* __restrict__ const& sQ,
    uint* __restrict__ const& rQ,
    unsigned int* __restrict__ const& pDB,
    const bool isMedley = false) {
    uint lRQIdx;
    // things are about to get warped :)
    // Aggregate tally across the warp
    WarpScan(wSt[0]).InclusiveSum(lTt, lRQIdx, taskTally);
    lRQIdx -= lTt;
    while (taskTally) {
        // Find processors if we are not currently aware of any
        while (!processorTally) {
            // sweep sQ to identify ready processes
            uint lPt = 0U; // local processor tally
            constexpr auto pL = processors / wS;
            #pragma unroll
            for (uint j = 0; j < pL; ++j) {
                const auto readiness = atomicExch(sQ + (j * wS + threadIdx.x),
                    observed) == ready;
                lPt += readiness;
                sQState[j] = readiness;
            }
            if (threadIdx.x < processors - pL * wS) {
                const auto readiness = atomicExch(sQ + (pL * wS + threadIdx.x),
                    observed) == ready;
                lPt += readiness;
                sQState[pL] = readiness;
            }
            uint startIdx;
            // Aggregate tally across the warp
            WarpScan(wSt[1]).InclusiveSum(lPt, startIdx, processorTally);
            startIdx -= lPt;
            // write to rQ
            if (lPt) {
                #pragma unroll
                for (uint j = 0; j < SQState::kElements; ++j) {
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
            if (isMedley) {
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
            for (uint j = sL; j < TQState::kElements; ++j) {
                if (tqState[j].tasks && tasksToSchedule) {
                    tqState[j].tasks = 0U;
                    const auto canSchedule = cute::min(tasksToSchedule, tqState[j].tasks);
                    tasksToSchedule -= canSchedule;
                    tqState[j].tasks -= canSchedule;
                    const auto taskIdx = tQRl * subscribers + (j - sL) * wS + threadIdx.x + tqState[j].tQTail;
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
void start(cuda::std::byte* __restrict__ workspace,
    const unsigned int& tQRl,
    const unsigned int& gtQCL,
    unsigned int* __restrict__ const& tQHeads, // shared
    unsigned int* __restrict__ const& gtQHeads, // global
    unsigned int* __restrict__ const& taskBound, // shared
    unsigned int* __restrict__ const& rQ, // shared
    unsigned int* __restrict__ const& sQ, // global
    unsigned int* __restrict__ const& pDB) { //  global
    uint scheduled = 0U;
    constexpr auto wS = 32U;
    constexpr auto sQsL = cute::ceil_div(processors, wS);
    static_assert(sQsL <= 32);

    constexpr auto subscribers = 128 - wS;
    static_assert(subscribers % wS == 0);
    constexpr auto sL = subscribers / wS;
    // initialize register buffers
    cutlass::Array<TQState, 16 + sL> tqState{};
    cutlass::Array<uint8_t, sQsL> sQState{};
    tqState.fill({0U,0U});
    sQState.fill(0U);

    constexpr auto dQL = decltype(tqState)::kElements - sL;
    const uint dT = gtQCL / (wS * dQL);

    // cub stuff
    using WarpScan = cub::WarpScan<uint>;
    auto* __restrict__ wSt = CAST_TO(WarpScan::TempStorage, workspace);
    uint gRQIdx = 0U;
    uint taskTally = 0U;
    uint processorTally = processors; // initially, all processors are available, ensure that rQ has all pids
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
                const auto tasks = atomicLoad(gtQHeads + qIdx) - tqState[j].tQTail;
                tqState[j].tasks = tasks;
                lTt += tasks;
            }
            // schedule observed tasks
            schedulerLoop<processors>(sQState, tqState, tQRl, lTt, taskTally,
                processorTally, gRQIdx, pTEpilog, scheduled, wSt, sQ, rQ, pDB, true);

            for (uint i = 1; i < dT; ++i) {
                // Needed to enforce register storage
                #pragma unroll
                for (uint j = sL; j < decltype(tqState)::kElements; ++j) {
                    const auto qIdx = wS * (dQL * i + (j - sL)) + threadIdx.x;
                    const auto tasks = atomicLoad(gtQHeads + qIdx) - tqState[j].tQTail;
                    tqState[j].tasks = tasks;
                    lTt += tasks;
                }
                // schedule observed tasks
                schedulerLoop<processors>(sQState, tqState, tQRl, lTt, taskTally,
                    processorTally, gRQIdx, pTEpilog, scheduled, wSt, sQ, rQ, pDB);
            }
        }

        // residue
        #pragma unroll
        for (uint j = sL; j < decltype(tqState)::kElements; ++j) {
            if (const auto qIdx = wS * (dQL * dT + (j - sL)) + threadIdx.x; qIdx < gtQCL) {
                const auto tasks = atomicLoad(gtQHeads + qIdx) - tqState[j].tQTail;
                tqState[j].tasks = tasks;
                lTt += tasks;
            }
        }
        // schedule observed tasks
        schedulerLoop<processors>(sQState, tqState, tQRl, lTt, taskTally,
            processorTally, gRQIdx, pTEpilog, scheduled, wSt, sQ, rQ, pDB, dT == 0);
    }
}

template<
    unsigned int processorCount,
    unsigned int M,
    unsigned int N,
    unsigned int Nx,
    unsigned int bM,
    unsigned int bN,
    unsigned int threads
>
requires(M % 64 == 0 && Nx % bN == 0 && N % bN == 0 && processorCount > 0)
__global__ __maxnreg__(128) void wScheduler(unsigned int* __restrict__ p, const bool skip = true) {
    constexpr auto wS = 32;
    constexpr auto subscribers = threads - wS;
    constexpr auto tilesM = M / bM;
    constexpr auto tilesN = N / bN;
    constexpr auto tilesNx = Nx / bN;
    constexpr auto gtQCL = tilesM;
    constexpr auto nTQH = subscribers + gtQCL;
    constexpr auto fTB = tilesM * tilesNx;
    constexpr auto tQRl = cute::ceil_div(fTB, subscribers);
    auto* __restrict__ interrupt = p;
    if (blockIdx.x + 1 == processorCount) {
        constexpr auto scratchSize = (nTQH + 1 + processorCount) * sizeof(uint) +
            2 * sizeof(cub::WarpScan<uint>::TempStorage);
        __shared__ __align__(16) cuda::std::byte workspace[scratchSize];
        auto* __restrict__ gtQHeads = interrupt + processorCount;
        auto* __restrict__ sQ = gtQHeads + gtQCL;
        #pragma unroll
        for (uint i = threadIdx.x; i < gtQCL; i += threads) {
            gtQHeads[i] = tilesN;
        }
        auto* __restrict__ tQHeads = CAST_TO(uint, workspace);
        constexpr auto residue = fTB - fTB / subscribers * subscribers;
        #pragma unroll
        for (uint i = threadIdx.x; i < nTQH; i+= threads) {
            tQHeads[i] = fTB / subscribers + (i < residue);
        }
        auto* __restrict__ rQ = tQHeads + nTQH;
        #pragma unroll
        for (uint i = threadIdx.x; i < processorCount; i+= threads) {
            rQ[i] = i;
        }
        auto* __restrict__ taskBound = rQ + 1;
        if (!threadIdx.x) {
            *taskBound = tilesM * tilesN + tilesM * tilesNx;
        }
        __syncthreads();
        if (threadIdx.x % wS == 0) {
            uint64_t begin, end;
            asm volatile("mov.u64 %0, %%globaltimer;": "=l"(begin)::);
            start<processorCount>(workspace + (nTQH + 1 + processorCount) * sizeof(uint), tQRl, gtQCL, tQHeads,
                gtQHeads, taskBound, rQ, sQ, sQ + processorCount);
            asm volatile("mov.u64 %0, %%globaltimer;": "=l"(end)::);
            __syncwarp();
            if(!skip && !threadIdx.x) {
                printf("Time taken is %fus\n", static_cast<float>(end - begin) / 1000.0f);
            }
        }
        __syncthreads();
        #pragma unroll
        for (uint i = threadIdx.x; i < processorCount; i += threads) {
            atomicInc(interrupt + i, UINT_MAX);
        }
    }
    else {
        auto* __restrict__ sQ = interrupt + processorCount + gtQCL;
        if (const auto idx = threads * blockIdx.x + threadIdx.x; idx < processorCount) {
            while (!atomicLoad(interrupt + idx)) {
                atomicExch(sQ + idx, ready);
            }
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

    constexpr auto processorCount = 4 * 108U; // Ampere
    constexpr auto threads = 128U;
    constexpr auto gtQCl = M / bM;
    constexpr auto len = (3 * processorCount + gtQCl) * sizeof(unsigned int);

    unsigned int* p;
    CHECK_ERROR_EXIT(cudaMallocAsync(&p, len, cudaStreamPerThread));
    CHECK_ERROR_EXIT(cudaMemsetAsync(p, 0U, processorCount * sizeof(uint), cudaStreamPerThread));

    for (uint i = 0; i < 256; ++i) {
        wScheduler<processorCount, M, N, Nx, bM, bN, threads><<<processorCount, threads, 0, cudaStreamPerThread>>>(p);
        CHECK_ERROR_EXIT(cudaMemsetAsync(p, 0U, processorCount * sizeof(uint), cudaStreamPerThread));
    }
    wScheduler<processorCount, M, N, Nx, bM, bN, threads><<<processorCount, threads, 0, cudaStreamPerThread>>>(p, false);
    CHECK_ERROR_EXIT(cudaFreeAsync(p, cudaStreamPerThread));
    CHECK_LAST();
}
#endif //WARPSCHEDULER_CUH
