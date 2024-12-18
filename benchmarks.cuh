//
// Created by oja7 on 12/18/24.
//

#ifndef BENCHMARKS_CUH
#define BENCHMARKS_CUH

#include <array>

#include <cooperative_groups/memcpy_async.h>
#include <cuda/barrier>
#include <cute/tensor.hpp>
#include <cutlass/array.h>
#include "util.cuh"
#include "processor/tiling.cuh"

__device__ __inline__ unsigned int q[3];
constexpr unsigned int bb = 4096;
__device__ __inline__ unsigned int qq[bb];
__device__ __inline__ unsigned int pDB[bb];
template<unsigned int p>
__global__ void testScheduler(unsigned int bound, bool skip = true) {
    if (threadIdx.x == 0) {
        __shared__ unsigned int up;
        __shared__ unsigned int rQ[p];
        #pragma unroll
        for (int i = 0; i < p; ++i) {
            rQ[i] = i;
        }
        up = bound;
        unsigned int sX = 0U;
        unsigned int tail = 0U;
        size_t start = 0, end = 0;
        q[0] = bound;
        q[1] = bound;
        q[2] = 0U;
        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(start)::);
        while (sX < atomicOr_block(&up, 0U)) {
            auto x = atomicOr(q, 0U) - sX;
            while (x > 0) {
                auto y = atomicOr(q + 1, 0U) - tail;
                while ( y > 0 && x > 0) {
                    auto r = qq[tail];
                    ++sX;
                    --x;
                    ++tail;
                    --y;
                    atomicExch(pDB + r, sX);
                }
            }
            // while (atomicOr(q, 0U) > sX && atomicOr(q + 1, 0U) > tail) {
            //     ++sX;
            //     ++tail;
            // }
        }
        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(end)::);
        if (!skip) {
            printf("single takes: %f\n", static_cast<float>(end - start) / 1e3);
        }

    }
}

__global__ void benchAtAdd(const unsigned int __grid_constant__ bound, bool skip = true) {
    if (!threadIdx.x) {
        q[0] = 0U;
    }
    __syncthreads();
    float d = 0.0f, d2 = 0.0f;
    for (int i = 0; i < bound; ++i) {
        size_t start = 0, end = 0;
        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(start)::);
        atomicAdd(q, 1U);
        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(end)::);
        d += static_cast<float>(end - start) / static_cast<float>(bound);
        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(start)::);
        atomicInc(q, UINT32_MAX);
        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(end)::);
        d2 += static_cast<float>(end - start) / static_cast<float>(bound);
    }
    __syncthreads();
    if (!threadIdx.x && !skip) {
        printf("atAdd takes: %f, atInc takes: %f, val is %u, *val++ is %u\n", d, d2, *q, ++*q);
        *q += 1;
        printf("val: %u", *q);
    }
}

__host__ __forceinline__
void hostBenchAtAdd() {
    volatile unsigned int b = 4096;
    assert(b == bb);
    for (int i = 0; i < 128; ++i) {
        benchAtAdd<<<1,864>>>(b);
    }
    benchAtAdd<<<1,864>>>(b, false);
    CHECK_LAST();
}

__host__ __forceinline__
void hostSc() {
    CHECK_ERROR_EXIT(cudaSetDevice(1));
    volatile unsigned int b = 4096;
    for (int i = 0; i < 128; ++i) {
        testScheduler<bb><<<1,1>>>(b);
    }
    testScheduler<bb><<<1,1>>>(b, false);
    CHECK_LAST();
}

struct FooBarrier {
    cuda::barrier<cuda::thread_scope_device>* deviceBarrier;
    __forceinline__ __device__
    FooBarrier() = default;

    __host__ __forceinline__
    explicit FooBarrier(cuda::barrier<cuda::thread_scope_device>* _deviceBarrier):
    deviceBarrier(_deviceBarrier){}
};

__constant__ __inline__ FooBarrier fDevice;
template<unsigned int blocks>
__global__ void gridBarrier(cuda::barrier<cuda::thread_scope_device>* deviceBarrier, bool skip = true) {
    float d = 0.0f;
    constexpr auto rounds = 64;
    for (unsigned int i = 0; i < rounds; ++i) {
        size_t start = 0, end = 0;
        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(start)::);
        if (!threadIdx.x) {
            if ((atomicAdd(q, 1U) + 1) % blocks == 0) {
                atomicAdd(q + 1, 1U);
            }
            while (atomicOr(q + 1, 0U) != i + 1){}
        }
        __syncthreads();
        /*if (!threadIdx.x) {
            fDevice.deviceBarrier->arrive_and_wait();
        }
        __syncthreads();*/
        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(end)::);
        d += static_cast<float>(end - start) / static_cast<float>(rounds);
    }
    if (!skip && !threadIdx.x) {
        printf("Block %u says Blockade takes: %f and q is %u\n", blockIdx.x, d, *q);
    }
}

__host__ __forceinline__
void hostB() {
    CHECK_ERROR_EXIT(cudaSetDevice(1));
    constexpr std::array<unsigned int, 2> qHost{{0U, 0U}};
    CHECK_ERROR_EXIT(cudaMemcpyToSymbol(q, qHost.data(), qHost.size()*sizeof(unsigned int)));
    const auto host_b = new cuda::barrier<cuda::thread_scope_device>{256};
    cuda::barrier<cuda::thread_scope_device>* b;
    CHECK_ERROR_EXIT(cudaMalloc(&b, sizeof(cuda::barrier<cuda::thread_scope_device>)));
    CHECK_LAST();
    CHECK_ERROR_EXIT(cudaMemcpy(b, host_b, sizeof(cuda::barrier<cuda::thread_scope_device>), cudaMemcpyHostToDevice));
    CHECK_LAST();

    const auto f = FooBarrier(b);
    CHECK_ERROR_EXIT(cudaMemcpyToSymbol(fDevice, &f, sizeof(cuda::barrier<cuda::thread_scope_device>)));
    gridBarrier<256><<<256, 128>>>(b, false);
    CHECK_LAST();
    delete host_b;
}

__global__ void expBench() {
    constexpr auto k = 64;
    cutlass::AlignedArray<float, cute::max(k, 32)> rScratch{};
    __shared__ float gateScratch[128];
    if (!threadIdx.x) {
        #pragma unroll
        for (uint i = 0; i < 128; ++i) {
            gateScratch[i] = static_cast<float>(i);
        }
    }
    __syncthreads();

    /*using sCLay = cute::Layout<cute::Shape<cute::Int<64>, cute::Int<2>>>;*/
    using sCLayX = cute::Layout<cute::Shape<cute::Int<64>, cute::Int<2>>, cute::Stride<cute::_2, cute::_1>>;
    constexpr auto l = make_layout(cute::Shape<cute::Int<64>, cute::Int<2>>{}, cute::LayoutRight{});
    /*using sCLay2 = cute::Layout<cute::Shape<cute::Int<2>, cute::Int<64>>>;
    using sCLay2X = cute::Layout<cute::Shape<cute::Int<2>, cute::Int<64>>, cute::Stride<cute::_64, cute::_1>>;*/
    /*print_tensor(make_tensor(cute::make_smem_ptr(gateScratch), sCLay{}));
    printf("t(1): %f\n", make_tensor(cute::make_smem_ptr(gateScratch), sCLay{})(1));*/
    auto t = make_tensor(cute::make_smem_ptr(gateScratch), l);
    /*print_tensor(t);
    printf("t(1): %f\n", make_tensor(cute::make_smem_ptr(gateScratch), sCLayX{})(1));*/
    /*print_tensor(make_tensor(cute::make_smem_ptr(gateScratch), sCLay2{}));
    printf("t(1): %f\n", make_tensor(cute::make_smem_ptr(gateScratch), sCLay2{})(1));
    print_tensor(make_tensor(cute::make_smem_ptr(gateScratch), sCLay2X{}));
    printf("t(1): %f\n", make_tensor(cute::make_smem_ptr(gateScratch), sCLay2X{})(1));*/
}

template<typename BlockMM>
__global__ __maxnreg__(128) void regPair() {
    cutlass::AlignedArray<cuda::std::pair<float, unsigned int>, 32> f{};
    #pragma unroll
    for (int i = 0; i < f.size(); ++i) {
        const bool isRegister = !(__isShared(f.data() + i) &&
        __isLocal(f.data() + i) &&
        __isConstant(f.data() + i) &&
        __isGlobal(f.data() + i) &&
        __isGridConstant(f.data() + i));
        assert(isRegister);
    }
    assert(isLikelyRegister(&blockIdx.x));
    constexpr auto t = BlockMM::block_dim.x;
    static_assert(BlockMM::block_dim.x == 128);
}

__device__ unsigned int res = 0U;
__global__ void slowMod(unsigned int rounds) {
    auto z = 0U;
    float duration = 0.0f;
    for (int i = 0; i < rounds; ++i) {
        size_t start, end;
        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(start)::);
        z = i % gridDim.x;
        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(end)::);
        duration += static_cast<float>(end - start) / rounds;
    }
    atomicExch(&res, z);

    if (cooperative_groups::grid_group::block_rank() == 0) {
        printf("slowMod takes: %f\n", duration);
    }
}

template<unsigned int p>
__global__ void fastMod(unsigned int rounds) {
    auto x = 0U;
    float duration = 0.0f;
    for (unsigned int i = 0; i < rounds; ++i) {
        size_t start, end;
        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(start)::);
        x = i % p;
        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(end)::);
        duration += static_cast<float>(end - start) / rounds;
    }
    atomicExch(&res, x);
    if (cooperative_groups::grid_group::block_rank() == 0) {
        printf("FastMod takes: %f\n", duration);
    }

}

void testModHost() {
    volatile int x = 800;
    volatile uint rounds = 1024;
    slowMod<<<x, 1>>>(rounds);
    CHECK_LAST();
}

__constant__ __inline__ unsigned int x = 80U;
template<typename T>
__global__ __maxnreg__(32) void benchContention(T* p, const __grid_constant__ T top) {
    if (cooperative_groups::thread_block::thread_rank() == 0) {
        constexpr auto rounds = 1024U;
        float duration = 0.0;
        __shared__ float scratch;
        scratch = 0.0f;
        for (int i = 0; i < rounds; ++i) {
            size_t start, end;
            asm volatile("mov.u64 %0, %%globaltimer;": "=l"(start)::);
            atomicExch(p, 23UL);
            asm volatile("mov.u64 %0, %%globaltimer;": "=l"(end)::);
            duration += static_cast<float>(end - start) / rounds;
        }
        cuda::std::ignore = cuda::atomic_ref{scratch}.fetch_max(duration);
        printf("Block %lu: Time to do work is %f and x is %u\n",
            cooperative_groups::grid_group::block_rank(), scratch, x);
    }
}

__host__ __forceinline__
void persisting() {
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "L2 Cache: " << prop.l2CacheSize << " L2 Persist Window: " << prop.accessPolicyMaxWindowSize << std::endl;
    int dev = 0, devAttr = 0;
    CUTE_CHECK_ERROR(cudaDeviceGetAttribute(&devAttr, cudaDevAttrL2CacheSize, dev));
}

__global__ void benchPersist(unsigned int* p) {
    constexpr auto rounds = 1024U;
    const auto tid = cooperative_groups::thread_block::thread_rank();
    float duration = 0.0;
    for (int i = 0; i < rounds; ++i) {
        size_t start, end;
        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(start)::);
        for (unsigned int j = 0; j < 128 * 128; j += 128) {
            auto x = p[j];
            p[j] = tid + static_cast<unsigned int>(x * sinf(static_cast<float>(tid)));
        }
        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(end)::);
        duration += static_cast<float>(end - start) / (static_cast<float>(NANO_TO_MICRO) * rounds);
    }
    __syncthreads();
    auto result = CAST_TO(float, p);
    result[0] = 0.0;
    cuda::std::ignore = cuda::atomic_ref{result[0]}.fetch_max(duration);
    __syncthreads();
    if (tid == 0) {
        printf("Time to do work is %f\n", result[0]);
        result[0] = 0.0;
    }
}

__host__ __forceinline__
void streamPersist(void* p, const unsigned long& bytes) {
    cudaStreamAttrValue stream_attribute;   // Stream level attributes data structure
    stream_attribute.accessPolicyWindow.base_ptr  = p; // Global Memory data pointer
    // Number of bytes for persistence access.
    stream_attribute.accessPolicyWindow.num_bytes = bytes;
    // (Must be less than cudaDeviceProp::accessPolicyMaxWindowSize)
    stream_attribute.accessPolicyWindow.hitRatio  = 1.0;                          // Hint for cache hit ratio
    stream_attribute.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting; // Type of access property on cache hit
    stream_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;  // Type of access property on cache miss.

    //Set the attributes to a CUDA stream of type cudaStream_t
    cudaStreamSetAttribute(cudaStreamPerThread, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);
}

__host__ __forceinline__
void benchContentionHost() {
    using at = unsigned long long int;
    constexpr at t = (431 * 1024) + 1;
    void* p;
    CUTE_CHECK_ERROR(cudaMalloc(&p, sizeof(at)));
    CUTE_CHECK_ERROR(cudaMemset(p, 0, sizeof(at)));
    benchContention<<<431, 128>>>(static_cast<at*>(p), t);
    CUTE_CHECK_ERROR(cudaMemset(p, 0, sizeof(at)));
    CUTE_CHECK_ERROR(cudaDeviceSynchronize());
    printf("--------------------------------------------------\n");
    streamPersist(p, sizeof(at));
    benchContention<<<431, 128, 0, cudaStreamPerThread>>>(static_cast<at *>(p), t);
    CUTE_CHECK_LAST();
    at x;
    CUTE_CHECK_ERROR(cudaMemcpy(&x, p, sizeof(at), cudaMemcpyDeviceToHost));
    printf("p is %lu", x);
}

void __global__ benchShared(float in) {
    extern __shared__ cuda::std::byte pad[];
    bool* interrupt = CAST_TO(bool, pad);
    if (cooperative_groups::thread_block::thread_rank() == 0) {
        *interrupt = false;
    }
    for (unsigned int i = threadIdx.x; i < 4096; i += 128) {
        CAST_TO(float, pad)[i] = 0.0f;
    }
    __syncthreads();

    while (!*interrupt) {
        for (unsigned int i = threadIdx.x; i < 4096; i += 128) {
            CAST_TO(float, pad)[i] += 0.1f;
        }
        if (cooperative_groups::thread_block::thread_rank() == 0) {
            interrupt[0] = CAST_TO(float, pad)[0] > in;
        }
        __syncthreads();
    }
}

__global__ __maxnreg__(128) void atoEx(unsigned int* p) {
    if (blockIdx.x == 0) {
        // producer
        for (int i = 0; i < 8; ++i) {
            for (unsigned int j = threadIdx.x; j < 401; j += 128) {
                atomicExch(p + j, i + 1);
            }
        }
    }
    else {
        if (threadIdx.x == 0) {
            // consumer
            float durationEx = 0.0;
            for (int i = 0; i < 8; ++i) {
                size_t start, end;
                unsigned int x = 0U, next=0U;
                asm volatile("mov.u64 %0, %%globaltimer;": "=l"(start)::);
                // pass on
                while (atomicOr(p + blockIdx.x, 0U) == i) {}
                asm volatile("mov.u64 %0, %%globaltimer;": "=l"(end)::);
                durationEx += static_cast<float>(end - start) / 8.0f;
            }
            printf("Block %u, V: %u, AtoEx Val: %f\n", blockIdx.x, atomicOr(p + blockIdx.x, 0U), durationEx);
        }
    }
}

void atoExHost() {
    void* p;
    constexpr std::array<unsigned int, 400> arr{};
    CUTE_CHECK_ERROR(cudaMallocAsync(&p, sizeof(unsigned int)*arr.size(), cudaStreamPerThread));
    CUTE_CHECK_ERROR(cudaMemsetAsync(p, 0, sizeof(unsigned int)*arr.size(),
        cudaStreamPerThread));
    atoEx<<<401,128,0,cudaStreamPerThread>>>(static_cast<unsigned int*>(p));
    CUTE_CHECK_LAST();
}

template<unsigned int threads = 128, unsigned int buf = 4 * 1024>
requires(buf > threads && buf % threads == 0)
__global__ void benchBankConflict() {
    assert(blockDim.x * blockDim.y * blockDim.z == threads);
    size_t start = 0, end = 0;
    double freeTime = 0.0, blockedTime = 0.0;
    __shared__ unsigned int db[buf];
    constexpr auto elems = buf / threads;
    const unsigned int tid = cooperative_groups::thread_block::thread_rank();
    unsigned int x = 0;
    #pragma unroll
    for (int i = 0; i < 1024; ++i) {
        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(start)::);
        // Minimizes bank-free conflict
        #pragma unroll
        for (int j = 0; j < elems; ++j) {
            // Write
            db[tid + j * threads] = tid;
            // Read
            x += db[tid + j * threads];
        }
        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(end)::);
        freeTime += static_cast<double>(end - start) / 1024.0;

        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(start)::);
        // Blocked access
        #pragma unroll
        for (int j = 0; j < elems; ++j) {
            // Write
            db[j + tid * elems] = tid;
            // Read
            x += db[j + tid * elems];
        }
        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(end)::);
        blockedTime += static_cast<double>(end - start) / 1024.0;
    }
    if (tid == 0) {
        printf("Free: %f, Block: %f\n", freeTime, blockedTime);
    }
}

#define N_FOO 8
__device__ unsigned long int foo[N_FOO];
__global__ void testAtomicMax() {
    const unsigned int tid = cooperative_groups::thread_block::thread_rank();
    const unsigned int bid = cooperative_groups::grid_group::block_rank();
    const unsigned int nB = cooperative_groups::grid_group::num_blocks();
    #pragma unroll
    for (unsigned int i = tid; i < N_FOO; i += THREADS) {
        cuda::std::ignore = cuda::atomic_ref<unsigned long int, cuda::thread_scope_device>{foo[i]}.fetch_max(bid);
    }
    __syncthreads();
    if (bid == 0) {
        #pragma unroll
        for (unsigned int i = tid; i < N_FOO; i += THREADS) {
            if (foo[i] != nB - 1) {
                printf("foo[%u]: %lu is wrong\n", i, foo[i]);
                assert(false);
            }
        }
    }
}
void hostAMax() {
    auto* p = calloc(N_FOO, sizeof(unsigned long int));
    // Sets foo to zero
    CUTE_CHECK_ERROR(cudaMemcpyToSymbol(foo, p, N_FOO * sizeof(unsigned long int)));
    testAtomicMax<<<64, THREADS>>>();
    CUTE_CHECK_LAST();
    free(p);
}


__host__ __forceinline__
void testBiasTrick() {
    cute::array<float, 4> a{{0, 1, 2, 3}};
    auto t = make_tensor(cute::make_gmem_ptr(a.data()), make_layout(cute::make_shape(2,2), cute::LayoutRight{}));
    print_tensor(t);
    cute::array<float, 2> b{{4, 5}};
    auto bias = make_tensor(b.data(), make_layout(cute::make_shape(2,2), cute::make_stride(0, 1)));
    axpby(1.0f, bias, 1.0f, t);
    print_tensor(bias);
    print_tensor(t);
}
#endif //BENCHMARKS_CUH
