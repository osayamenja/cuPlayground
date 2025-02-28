#include <cuda/std/array>
#include <cuda/std/functional>

#include "mma.cuh"
#include "util.cuh"

__global__ __maxnreg__(255) void theatre(const float* p,
    const uint* q, float* r, const uint n) {
    __shared__ float workspace[8 * 1024];
    __shared__ uint spare[156];
    __shared__ void* ptrs[4];

    if (threadIdx.x < 4) {
        ptrs[threadIdx.x] = workspace + threadIdx.x;
    }
    for (uint i = threadIdx.x; i < n; i += blockDim.x) {
        workspace[i] = p[i];
    }
    spare[threadIdx.x] = q[threadIdx.x] * static_cast<uint>(p[threadIdx.x]);
    __syncthreads();
    for (uint i = threadIdx.x; i < n; i += blockDim.x) {
        r[i] = workspace[i] * static_cast<float>(spare[threadIdx.x]);
    }
    if (threadIdx.x < 4) {
        r[threadIdx.x] *= CAST_TO(float, ptrs[threadIdx.x])[threadIdx.x];
    }
}

__host__ __forceinline__
void stage() {
    void* p;
    CHECK_ERROR_EXIT(cudaMalloc(&p, sizeof(float) * 1536));
    uint n = 512;
    theatre<<<5 * 108, 128>>>(CAST_TO(float, p),
        CAST_TO(uint, p) + 512, CAST_TO(float, p) + 1024, n);
    CHECK_LAST();
}
int main() {
    testCollective();
}
