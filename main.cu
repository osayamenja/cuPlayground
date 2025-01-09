/*#include <nvshmemx.h>
#include <nvshmem.h>
#include <host/nvshmemx_api.h> // Makes CLion happy*/

#include <cuda/std/functional>
#include <cuda/std/__algorithm/make_heap.h>
#include <cuda/std/__algorithm/pop_heap.h>
#include <cuda/std/array>
#include <cute/tensor.hpp>
#include "util.cuh"

using V = float;
using HT = cuda::std::pair<uint, uint>;

template<unsigned int n, typename T>
__device__ __forceinline__
void insertionSort(T* __restrict__ const& a) {
    #pragma unroll
    for (uint i = 1; i < n; ++i) {
        #pragma unroll
        for (uint j = i; j > 0; --j) {
            if (a[j - 1] > a[j]) {
                cuda::std::swap(a[j - 1], a[j]);
            }
        }
    }
}

template<unsigned int n, typename T>
__device__ __forceinline__
void selectionSort(T* __restrict__ const& a) {
    #pragma unroll
    for (uint i = 0; i < n - 1; ++i) {
        auto jM = i;
        #pragma unroll
        for (uint j = i + 1; j < n; j++) {
            if (a[j] < a[jM]) {
                jM = j;
            }
        }
        cuda::std::swap(a[jM], a[i]);
    }
}

__global__ void theatre(const bool skip = true) {
    constexpr auto k = 4U;
    cuda::std::array<HT, k> heap{};
    __shared__ HT sHeap[k];
    #pragma unroll
    for (uint i = 0; i < k; ++i) {
        sHeap[i] = HT{k, i};
        heap[i] = HT{k, i};
    }
    uint64_t start = 0, end = 0;
    float sC = 0.0f;
    float lC = 0.0f;
    float qC = 0.0f;
    // print_tensor(heapT); printf("\n");
    asm volatile("mov.u64 %0, %%globaltimer;": "=l"(start)::);
    make_heap(sHeap, sHeap + k, cuda::std::greater{});
    pop_heap(sHeap, sHeap + k, cuda::std::greater{});
    #pragma unroll
    for (uint i = 0; i < 64 - k; ++i) {
        push_heap(sHeap, sHeap + k, cuda::std::greater{});
        pop_heap(sHeap, sHeap + k, cuda::std::greater{});
    }
    asm volatile("mov.u64 %0, %%globaltimer;": "=l"(end)::);
    sC = static_cast<float>(end - start);
    asm volatile("mov.u64 %0, %%globaltimer;": "=l"(start)::);
    make_heap(heap.begin(), heap.end(), cuda::std::greater{});
    pop_heap(heap.begin(), heap.end(), cuda::std::greater{});
    #pragma unroll
    for (uint i = 0; i < 64 - k; ++i) {
        push_heap(heap.begin(), heap.end(), cuda::std::greater{});
        pop_heap(heap.begin(), heap.end(), cuda::std::greater{});
    }
    asm volatile("mov.u64 %0, %%globaltimer;": "=l"(end)::);
    lC = static_cast<float>(end - start);
    asm volatile("mov.u64 %0, %%globaltimer;": "=l"(start)::);
    asm volatile("mov.u64 %0, %%globaltimer;": "=l"(end)::);
    qC = static_cast<float>(end - start);
    if (!skip) {
        printf("sC is %f, lC is %f, qC is %f", sC, lC, qC);
    }

}
#define TIMING 1

__global__ void stage(const float init, uint* __restrict__ p, const bool skip = true) {
    constexpr auto k = 64U;
    constexpr auto ik = 1U;
    cutlass::AlignedArray<float, k> heap{};
    cutlass::AlignedArray<uint8_t, k> rC{};
    __shared__ __align__(16) uint8_t checked[k];
    __shared__ __align__(16) uint8_t indices[ik];
    #pragma unroll
    for (uint i = 0; i < k; ++i) {
        heap[i] = init - static_cast<float>(i);
        checked[i] = 0U;
    }
    auto ii = 0U;
#if TIMING
    uint64_t start = 0, end = 0;
    double iC = 0.0f;
    asm volatile("mov.u64 %0, %%globaltimer;": "=l"(start)::);
#endif

    #pragma unroll
    for (uint i = 0; i < ik; ++i) {
        auto v = -cuda::std::numeric_limits<V>::infinity();
        uint idx = 0U;
        #pragma unroll
        for (uint j = 0; j < k; ++j) {
            rC[j] = checked[j];
        }
        #pragma unroll
        for (uint j = 0; j < k; ++j) {
            if (heap[j] > v && !rC[j]) {
                idx = j;
                v = heap[j];
            }
        }
        checked[idx] = 1U;
        indices[ii++] = idx;
    }
#if TIMING
    asm volatile("mov.u64 %0, %%globaltimer;": "=l"(end)::);
    iC = static_cast<double>(end - start);
    if (!skip) {
        printf("iC is %f\n", iC);
    }
#endif
    p[ik - 1] = indices[ik - 1];
}

__host__ __forceinline__
void hostTheatre() {
    for (uint i = 0; i < 128; ++i) {
        theatre<<<1,1>>>();
    }
    theatre<<<1,1>>>(false);
    CUTE_CHECK_LAST();
}

__host__ __forceinline__
void hostStage() {
    // use volatile to deactivate compiler optimizations
    const volatile float k = 64.f;
    uint* p;
    CHECK_ERROR_EXIT(cudaMalloc(&p, 2 * k * sizeof(uint)));
#if TIMING
    for (uint i = 0; i < 128; ++i) {
        stage<<<1,1>>>(k, p);
    }
#endif
    stage<<<1,1>>>(k, p, false);
    CUTE_CHECK_LAST();
}
int main() {
    hostStage();
}