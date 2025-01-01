#include <cub/cub.cuh>
#include "util.cuh"

template<unsigned int threads>
__global__ void theatre(unsigned int* __restrict__ p, const bool skip = true) {
    __shared__ unsigned int pS;
    cuda::atomic_ref aSp{pS};
    cuda::atomic_ref aGp{*p};
    if (!threadIdx.x) {
        pS = threadIdx.x;
    }
    __syncthreads();

    double myClocked[2] = {0.0, 0.0};
    double theirsClocked[2] = {0.0, 0.0};
    constexpr auto rounds = 64;
    for (uint i = 0; i < rounds; ++i) {
        uint64_t start = 0, end = 0;
        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(start)::);
        cuda::std::ignore = atomicLoad<cuda::thread_scope_block>(&pS);
        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(end)::);
        myClocked[0] += static_cast<double>(end - start) / static_cast<double>(rounds);

        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(start)::);
        cuda::std::ignore = aSp.load();
        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(end)::);
        theirsClocked[0] += static_cast<double>(end - start) / static_cast<double>(rounds);

        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(start)::);
        cuda::std::ignore = atomicLoad(p);
        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(end)::);
        myClocked[1] += static_cast<double>(end - start) / static_cast<double>(rounds);

        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(start)::);
        cuda::std::ignore = aGp.load();
        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(end)::);
        theirsClocked[1] += static_cast<double>(end - start) / static_cast<double>(rounds);
    }
    using BlockReduce = cub::BlockReduce<double, threads>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    myClocked[0] = BlockReduce(temp_storage).Reduce(myClocked[0], cub::Max());
    myClocked[1] = BlockReduce(temp_storage).Reduce(myClocked[1], cub::Max());
    theirsClocked[0] = BlockReduce(temp_storage).Reduce(theirsClocked[0], cub::Max());
    theirsClocked[1] = BlockReduce(temp_storage).Reduce(theirsClocked[1], cub::Max());

    if (!threadIdx.x && !skip) {
        printf("Block %u, sAtomicLoad: %fns, gAtomicLoad: %fns, sCudaLoad: %fns, gCudaLoad: %fns\n",
            blockIdx.x, myClocked[0], myClocked[1], theirsClocked[0], theirsClocked[1]);
    }
}

__host__ __forceinline__
void hostTheatre() {
    unsigned int* p;
    CHECK_ERROR_EXIT(cudaMalloc(&p, sizeof(unsigned int)));
    constexpr auto threads = 128;
    /*for (uint i = 0; i < 128; ++i) {
        theatre<<<128, 128>>>(p);
    }*/
    theatre<threads><<<128, threads>>>(p);
    CHECK_LAST();
}

int main() {
    hostTheatre();
}