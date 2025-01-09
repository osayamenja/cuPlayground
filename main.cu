/*#include <nvshmemx.h>
#include <nvshmem.h>
#include <host/nvshmemx_api.h> // Makes CLion happy*/

#include <cuda/std/functional>
#include <cuda/std/__algorithm/make_heap.h>
#include <cuda/std/__algorithm/pop_heap.h>
#include <cuda/std/__algorithm/partial_sort.h>
#include <cuda/std/array>
#include <cute/tensor.hpp>
#include <cutlass/array.h>
#include "util.cuh"

using V = float;
using HT = cuda::std::pair<V, uint>;

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
/*__global__ void theatre(const bool skip = true) {
    constexpr auto k = 4U;
    cuda::std::array<V, k> heap{};
    __shared__ V sHeap[k];
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
    #pragma unroll
    for (uint i = 0; i < 64 - k; ++i) {
        pop_heap(sHeap, sHeap + k, cuda::std::greater{});
        push_heap(sHeap, sHeap + k, cuda::std::greater{});
    }
    asm volatile("mov.u64 %0, %%globaltimer;": "=l"(end)::);
    sC = static_cast<float>(end - start);

    asm volatile("mov.u64 %0, %%globaltimer;": "=l"(start)::);
    make_heap(heap.begin(), heap.end(), cuda::std::greater{});
    #pragma unroll
    for (uint i = 0; i < 64 - k; ++i) {
        pop_heap(heap.begin(), heap.end(), cuda::std::greater{});
        push_heap(heap.begin(), heap.end(), cuda::std::greater{});
    }
    asm volatile("mov.u64 %0, %%globaltimer;": "=l"(end)::);
    lC = static_cast<float>(end - start);

    asm volatile("mov.u64 %0, %%globaltimer;": "=l"(start)::);
    asm volatile("mov.u64 %0, %%globaltimer;": "=l"(end)::);
    qC = static_cast<float>(end - start);
    if (!skip) {
        printf("sC is %f, lC is %f, qC is %f", sC, lC, qC);
        cuda::std::array<uint, 4> a{4, 3, 2, 1};
        const auto t = make_tensor(a.data(), cute::Layout<cute::Shape<cute::_1, cute::_4>,
            cute::Stride<cute::_4, cute::_1>>{});
        print_tensor(t); printf("\n");
        selectionSort<4>(a.data());
        print_tensor(t); printf("\n");
    }
}*/

__global__ void stage(float* __restrict__ p, const bool skip = true) {
    constexpr auto k = 64U;
    constexpr auto ik = 2U;
    cuda::std::array<float, k> heap{};
    __shared__ bool checked[k];
    __shared__ uint indices[ik];
    #pragma unroll
    for (uint i = 0; i < k; ++i) {
        heap[i] = p[i];
    }

    auto ii = 0U;
    #pragma unroll
    for (uint i = 0; i < ik; ++i) {
        auto v = -cuda::std::numeric_limits<V>::infinity();
        auto idx = 0U;
        #pragma unroll
        for (uint j = 0; j < k; ++j) {
            if (heap[j] > v && !checked[j]) {
                idx = j;
                v = heap[j];
            }
        }
        checked[idx] = true;
        indices[ii++] = idx;
    }
    p[127] = static_cast<float>(indices[ik - 1]);
    /*uint64_t start = 0, end = 0;
    float iC = 0.0f;
    float sC = 0.0f;*/
    /*asm volatile("mov.u64 %0, %%globaltimer;": "=l"(start)::);*/
    //insertionSort<k>(heap.data());
    /*asm volatile("mov.u64 %0, %%globaltimer;": "=l"(end)::);
    iC = static_cast<float>(end - start);*/
    /*#pragma unroll
    for (uint i = 0; i < k; ++i) {
        p[i] = heap[i];
    }*/
    /*#pragma unroll
    for (uint i = 0; i < k; ++i) {
        heap[i] = p[k + i];
    }
    /*asm volatile("mov.u64 %0, %%globaltimer;": "=l"(start)::);#1#
    selectionSort<k>(heap.data());
    /*asm volatile("mov.u64 %0, %%globaltimer;": "=l"(end)::);
    sC = static_cast<float>(end - start);#1#
    #pragma unroll
    for (uint i = 0; i < k; ++i) {
        p[k + i] = heap[i];
    }*/
    /*if (!skip) {
        printf("sC is %f, iC is %f\n", sC, iC);
    }*/
}

__host__ __forceinline__
void hostTheatre() {
    /*for (uint i = 0; i < 128; ++i) {
        stage<<<1,1>>>();
    }*/
    constexpr auto k = 64U;
    cuda::std::array<V, k> heap{};
    for (uint i = 0; i < k; ++i) {
        heap[i] = static_cast<float>(k - i);
    }
    float* p;
    CHECK_ERROR_EXIT(cudaMalloc(&p, 2 * k * sizeof(float)));
    CHECK_ERROR_EXIT(cudaMemcpy(p, heap.data(), 2 * k * sizeof(float), cudaMemcpyHostToDevice));
    stage<<<1,1>>>(p);
    CUTE_CHECK_LAST();
}
int main() {
    hostTheatre();
}