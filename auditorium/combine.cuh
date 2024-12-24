//
// Created by osayamen on 12/22/24.
//

#ifndef COMBINE_CUH
#define COMBINE_CUH

#include <cooperative_groups/memcpy_async.h>
#include <cublasdx.hpp>
#include <cuda/std/array>
#include <cuda/std/type_traits>
#include <cute/tensor.hpp>
#include <cutlass/array.h>
#include <cuda/atomic>

#include "../util.cuh"
#include "../processor/tiling.cuh"

/// A more apropos name would be "static storage" rather than registers.
template<class T>
struct isRegister : cuda::std::false_type {};

template<class T, int N, int Alignment>
struct isRegister<cutlass::AlignedArray<T, N, Alignment>> : cuda::std::true_type {};

template<class T, int N, bool RegisterSized>
struct isRegister<cutlass::Array<T, N, RegisterSized>> : cuda::std::true_type {};

template<class Engine, class Layout>
struct isRegister<cute::Tensor<Engine, Layout>> :
cuda::std::conditional_t<cute::is_rmem_v<cute::Tensor<Engine, Layout>>,
cuda::std::true_type, cuda::std::false_type> {};

template <class T>
constexpr bool isRegisterV = isRegister<T>::value;

// Vector atomic add
template<unsigned int Arch, typename Element = float>
requires SupportedArch<Arch> && TensorValueType<Element>
struct VAA {
    template<class Registers>
    requires isRegisterV<Registers> &&
        cuda::std::is_same_v<typename Registers::value_type, Element>
    __device__ __forceinline__
    void operator()(Element* __restrict__ const& gS, Registers registers) const {
        #pragma unroll
        for (uint i = 0; i < registers.size(); ++i) {
            atomicAdd(gS + i, registers(i));
        }
    }
};

// specialization for half-precision
template<unsigned int Arch>
struct VAA<Arch, cute::half_t> {
    template<class Registers>
    requires isRegisterV<Registers> &&
        cuda::std::is_same_v<typename Registers::value_type, cute::half_t>
    __device__ __forceinline__
    void operator()(cute::half_t* __restrict__ const& gS, Registers registers) const {
        using vType = cuda::std::conditional_t<registers.size() % 2 == 0, __half2, __half>;
        constexpr auto len = registers.size() / (sizeof(vType) / sizeof(__half));
        auto* __restrict__ gSv = CAST_TO(vType, gS);
        const auto* __restrict__ vRegs = CAST_TO(vType, registers.data());
        #pragma unroll
        for (uint i = 0; i < len; ++i) {
            atomicAdd(gSv + i, vRegs[i]);
        }
    }
};

// specialization for bfloat16
template<unsigned int Arch> requires(Arch >= 800)
struct VAA<Arch, cute::bfloat16_t> {
    template<class Registers>
    requires isRegisterV<Registers> &&
        cuda::std::is_same_v<typename Registers::value_type, cute::bfloat16_t>
    __device__ __forceinline__
    void operator()(cute::bfloat16_t* __restrict__ const& gS, Registers registers) const {
        using vType = cuda::std::conditional_t<registers.size() % 2 == 0, __nv_bfloat162, __nv_bfloat16>;
        constexpr auto len = registers.size() / (sizeof(vType) / sizeof(__half));
        auto* __restrict__ gSv = CAST_TO(vType, gS);
        const auto* __restrict__ vRegs = CAST_TO(vType, registers.data());
        #pragma unroll
        for (uint i = 0; i < len; ++i) {
            atomicAdd(gSv + i, vRegs[i]);
        }
    }
};

// specialization for float on Hopper
template<>
struct VAA<900, float> {
    template<class Registers>
    requires isRegisterV<Registers> &&
        cuda::std::is_same_v<typename Registers::value_type, float>
    __device__ __forceinline__
    void operator()(float* __restrict__ const& gS, Registers registers) const {
        static_assert(registers.size() % 2 == 0, "Register tensor does not vectorize");
        using vType = cuda::std::conditional_t<registers.size() % 4 == 0, float4,
            cuda::std::conditional_t<registers.size() % 2 == 0, float2, float>>;
        constexpr auto len = registers.size() / (sizeof(vType) / sizeof(float));
        auto* __restrict__ gSv = CAST_TO(vType, gS);
        const auto* __restrict__ vRegs = CAST_TO(vType, registers.data());
        #pragma unroll
        for (uint i = 0; i < len; ++i) {
            atomicAdd(gSv + i, vRegs[i]);
        }
    }
};

template<unsigned Arch> requires SupportedArch<Arch>
struct Combine {
    template<
        class Activations,
        class Registers,
        typename Element = typename Activations::value_type
    >
    requires(TensorValueType<Element> &&
        cute::is_tensor_v<Activations> && isRegisterV<Registers>)
    __device__ __forceinline__
    void operator()(Element* __restrict__ workspace,
        const unsigned int* __restrict__ tokenIndices,
        Registers registers,
        Element* __restrict__ inputs,
        Activations const& activations,
        const unsigned int& M,
        const unsigned int& N,
        const unsigned int& tileIdx,
        const unsigned int& tileSize) const {

        using BlockTiler = cute::Shape<cute::Int<BLOCK_M>, cute::Int<BLOCK_N>>;
        constexpr BlockTiler tiler{};
        constexpr auto bM = cute::get<0>(tiler);
        constexpr auto bN = cute::get<1>(tiler);
        constexpr auto threads = THREADS;
        static_assert(!(cuda::std::is_same_v<Element, cute::float_e4m3_t> &&
            cuda::std::is_same_v<Element, cute::float_e4m3_t>), "fp8 atomic addition is not available, "
                                                                "so no support for this operation yet");

        // Eagerly issue gmem read.
        auto tokenIdx = tokenIndices[threadIdx.x];
        // Row-major
        const auto mA = make_tensor(cute::make_gmem_ptr(inputs),
            make_layout(cute::make_shape(M, N), cute::LayoutRight{}));

        const auto tilesM = M / cute::get<0>(tiler);
        // We assert the below prior to this point
        const auto tilesN = N / cute::get<1>(tiler);

        const auto tileCoord = idx2crd(tileIdx, cute::Shape(tilesM, tilesN), cute::Stride(tilesN ,1));
        const auto ctaCoord = cute::make_coord(cute::get<0>(tileCoord), cute::get<1>(tileCoord));
        const auto gA = cute::local_tile(mA, tiler, ctaCoord);
        constexpr auto elems = SHARED_SIZE / (threads * sizeof(Element));
        static_assert(bN % elems == 0);
        constexpr auto trips = bN / elems;

        // Transposed layout
        constexpr auto sCLay = cute::make_layout(cute::Shape<cute::Int<bM>, cute::Int<elems>>{}, cute::LayoutRight{});
        const auto sC = cute::make_tensor(cute::make_smem_ptr(workspace), sCLay);
        constexpr VAA<Arch, Element> vaa{};

        #pragma unroll
        for (uint i = 0; i < trips; ++i) {
            // global -> shared
            #pragma unroll
            for (uint j = 0; j < elems; ++j) {
                const auto rIdx = j + threadIdx.x / elems * elems;
                const auto cIdx =  threadIdx.x % elems + i * elems;
                sC(rIdx, cIdx) = gA(rIdx, cIdx);
            }
            __syncthreads();
            #pragma unroll
            for (uint j = 0; j < elems; ++j) {
                registers[j + i * elems] = sC(threadIdx.x, j);
            }
        }

        if (threadIdx.x < tileSize) {
            vaa(&activations(tokenIdx, 0), registers);
        }
    }
};

template<unsigned int M, unsigned int N, typename Element>
__global__ __maxnreg__(128) void deviceCombine(cuda::std::byte* __restrict__ p) {
    const auto* __restrict__ tokenIndices = CAST_TO(unsigned int, p);
    auto* __restrict__ inputs = CAST_TO(Element, p + sizeof(unsigned int) * M);
    const auto inT = make_tensor(
        cute::make_gmem_ptr(inputs),
        make_layout(cute::make_shape(M, N), cute::LayoutRight{}));
    const auto activations = make_tensor(
        cute::make_gmem_ptr(CAST_TO(Element, p + sizeof(unsigned int) * M + sizeof(Element) * M * N)),
        make_layout(cute::make_shape(M, N), cute::LayoutRight{}));
    static_assert(cuda::std::is_same_v<typename decltype(activations)::value_type, Element>);
    constexpr Combine<800> combineOp{};
    extern __shared__ __align__(16) Element scratch[];
    cutlass::AlignedArray<Element, BLOCK_N> regs{};
    cutlass::AlignedArray<cute::half_t, BLOCK_N> r{};
    cuda::std::array<cute::bfloat16_t, BLOCK_N> rr{};
    combineOp(scratch, tokenIndices, regs, inputs, activations, M, N, blockIdx.x, M);
    /*__syncthreads();
    if (!threadIdx.x) {
        cute::print_tensor(activations);
    }*/
}

__host__ __forceinline__
void hostCombine() {
    const auto playStream = cudaStreamPerThread;
    constexpr auto M = 128;
    constexpr auto N = 64;
    using inputValueType = cute::half_t;

    constexpr auto aSize = sizeof(inputValueType) * M * N;
    constexpr auto len = 2 * aSize + M * sizeof(unsigned int);
    cuda::std::byte* a;
    CHECK_ERROR_EXIT(cudaMallocAsync(&a, len, playStream));
    auto* data = static_cast<cuda::std::byte*>(calloc(len, sizeof(cuda::std::byte)));
    auto* indices = CAST_TO(unsigned int, data);
    for (uint i = 0; i < M; ++i) {
        indices[i] = i;
    }
    indices += M;
    auto* inputs = CAST_TO(inputValueType, indices);

    // inputs
    for (uint i = 0; i < M; ++i) {
        for (uint j = 0; j < N; ++j) {
            inputs[j + i * N] = inputValueType{i + 1};
        }
    }
    CHECK_ERROR_EXIT(cudaMemsetAsync(a, 0, aSize, playStream));
    CHECK_ERROR_EXIT(cudaMemcpyAsync(a, data, aSize + sizeof(unsigned int) * M, cudaMemcpyHostToDevice, playStream));
    deviceCombine<M, N, inputValueType><<<1, 128, SHARED_SIZE, playStream>>>(a);
    CHECK_ERROR_EXIT(cudaFreeAsync(a, playStream));
    free(data);
    CHECK_LAST();
}
#endif //COMBINE_CUH
