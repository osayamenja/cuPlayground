//
// Created by osayamen on 12/22/24.
//

#ifndef COMBINE_CUH
#define COMBINE_CUH

#include <cublasdx.hpp>
#include <cuda/std/type_traits>
#include <cute/tensor.hpp>

#include "../util.cuh"
#include "../processor/tiling.cuh"

enum class Modality{
    coalesce,
    nonCoalesce
};
template<unsigned int Arch, Modality m = Modality::coalesce>
requires(Arch >= 700 && Arch <= 900)
struct Combine {
    static_assert(m == Modality::coalesce);
    // default will be non-vectorized
    template<typename BlockGEMM, class Registers, class RegisterScratch, class Activations>
    requires TensorValueType<typename Activations::value_type>
    __device__ __forceinline__
    void operator()(typename BlockGEMM::MatrixAType* __restrict__ const& workspace,
        const unsigned int* __restrict__ tokenIndices,
        Registers& registers,
        RegisterScratch& rScratch,
        const typename BlockGEMM::MatrixAType* __restrict__ const& inputs,
        Activations const& activations,
        const unsigned int& M,
        const unsigned int& K,
        const unsigned int& tileIdx,
        const unsigned int& tileSize) const{
        using Element = typename Activations::value_type;
        static_assert(!(cuda::std::is_same_v<Element, cute::float_e4m3_t> &&
            cuda::std::is_same_v<Element, cute::float_e4m3_t>), "fp8 atomic addition is not available, "
                                                                "so no support for this operation yet");
        // assert(__isShared(tokenIndices))
        constexpr auto bN = cute::get<1>(typename BlockGEMM::BlockTiler{});
        using ElementSource = typename BlockGEMM::MatrixAType;
        static_assert(cute::is_rmem_v<Registers> && cute::size(registers) % rScratch.size() == 0);
        cute::clear(registers);

        // Row-major
        const auto mA = make_tensor(cute::make_gmem_ptr(inputs),
            make_layout(cute::make_shape(M, K), cute::make_stride(K, 1)));

        const auto tilesM = M / cute::get<0>(typename BlockGEMM::BlockTiler{});
        // We assert the below prior to this point
        const auto tilesK = K / cute::get<1>(typename BlockGEMM::BlockTiler{});

        const auto tileCoord = idx2crd(tileIdx, cute::Shape(tilesM, tilesK), cute::Stride(tilesK ,1));
        const auto ctaCoord = make_coord(cute::get<0>(tileCoord), cute::get<1>(tileCoord), cute::_);
        const auto gA = cute::local_tile(mA, typename BlockGEMM::BlockTiler{}, ctaCoord,
            cute::Step<cute::_1, cute::_1, cute::X>{});
        constexpr auto bM = cute::get<0>(typename BlockGEMM::BlockTiler{});
        constexpr auto threads = BlockGEMM::GEMM::block_dim.x;
        constexpr auto elems = SHARED_SIZE / (threads * sizeof(ElementSource));
        static_assert(rScratch.size() == elems);
        static_assert(bN % elems == 0);
        constexpr auto trips = bN / elems;

        // Transposed layout
        constexpr auto sCLay = cute::make_layout(cute::Shape<cute::Int<bM>, cute::Int<elems>>{}, cute::LayoutRight{});
        const auto sC = cute::make_tensor(cute::make_smem_ptr(CAST_TO(ElementSource, workspace)), sCLay);

        #pragma unroll
        for (uint i = 0; i < trips; ++i) {
            // global -> shared
            #pragma unroll
            for (uint j = 0; j < elems; ++j) {
                const auto rIdx = j + threadIdx.x / elems * elems;
                const auto cIdx =  threadIdx.x % elems + i * elems;
                sC(rIdx, cIdx) = gA(rIdx, cIdx);
            }
            // No barrier needed! This is because each thread copies the slice that it subsequently needs
            // shared -> register in parallel with global -> shared
            #pragma unroll
            for (uint j = 0; j < elems; ++j) {
                const auto rIdx = j + threadIdx.x / elems * elems;
                const auto cIdx =  threadIdx.x % elems + i * elems;
                const auto cIdxNext = threadIdx.x % elems + (i + 1) * elems;
                registers(j + i * elems) = sC(rIdx, cIdx);
                if (i + 1 < trips) {
                    sC(rIdx, cIdxNext) = gA(rIdx, cIdxNext);
                }
            }
        }
        const auto sliceIdx = threadIdx.x / elems;
        const auto slices = tileSize / elems;
        if (slices >= sliceIdx) {
            if (slices > sliceIdx) {
                #pragma unroll
                for (uint i = 0; i < trips; ++i) {
                    #pragma unroll
                    for (uint j = 0; j < elems; ++j) {
                        rScratch[j] = tokenIndices[j + i * elems];
                    }

                    #pragma unroll
                    for (uint j = 0; j < elems; ++j) {
                        const auto cIdx = threadIdx.x % elems + i * elems;
                        // assume activations have been cleared
                        atomicAdd(&activations(rScratch[j], cIdx), registers(j + i * elems));
                    }
                }
            }
            else {
                const auto residue = tileSize - slices * elems;
                #pragma unroll
                for (uint i = 0; i < trips; ++i) {
                    #pragma unroll
                    for (uint j = 0; j < residue; ++j) {
                        rScratch[j] = tokenIndices[j + i * residue];
                    }

                    #pragma unroll
                    for (uint j = 0; j < elems; ++j) {
                        const auto cIdx = threadIdx.x % elems + i * elems;
                        // assume activations have been cleared
                        atomicAdd(&activations(rScratch[j], cIdx), registers(j + i * elems));
                    }
                }
            }
        }
    }
};

template<unsigned Arch>
struct Combine<Arch, Modality::nonCoalesce> {
    template<typename BlockGEMM, class Registers, class RegisterScratch, class Activations>
    requires TensorValueType<typename Activations::value_type>
    __device__ __forceinline__
    void operator()(typename BlockGEMM::MatrixAType* __restrict__ const& workspace,
        const unsigned int* __restrict__ tokenIndices,
        Registers& registers,
        RegisterScratch& rScratch,
        const typename BlockGEMM::MatrixAType* __restrict__ const& inputs,
        Activations const& activations,
        const unsigned int& M,
        const unsigned int& K,
        const unsigned int& tileIdx,
        const unsigned int& tileSize) const{
        auto tokenIdx = tokenIndices[threadIdx.x];
        using Element = typename Activations::value_type;
        static_assert(!(cuda::std::is_same_v<Element, cute::float_e4m3_t> &&
            cuda::std::is_same_v<Element, cute::float_e4m3_t>), "fp8 atomic addition is not available, "
                                                                "so no support for this operation yet");
        // assert(__isShared(tokenIndices))
        constexpr auto bN = cute::get<1>(typename BlockGEMM::BlockTiler{});
        using ElementSource = typename BlockGEMM::MatrixAType;
        static_assert(cute::is_rmem_v<Registers> &&
            cute::size(registers) % rScratch.size() == 0 && cute::size(registers) == bN);

        // Row-major
        const auto mA = make_tensor(cute::make_gmem_ptr(inputs),
            make_layout(cute::make_shape(M, K), cute::make_stride(K, 1)));

        const auto tilesM = M / cute::get<0>(typename BlockGEMM::BlockTiler{});
        // We assert the below prior to this point
        const auto tilesK = K / cute::get<1>(typename BlockGEMM::BlockTiler{});

        const auto tileCoord = idx2crd(tileIdx, cute::Shape(tilesM, tilesK), cute::Stride(tilesK ,1));
        const auto ctaCoord = make_coord(cute::get<0>(tileCoord), cute::get<1>(tileCoord), cute::_);
        const auto gA = cute::local_tile(mA, typename BlockGEMM::BlockTiler{}, ctaCoord,
            cute::Step<cute::_1, cute::_1, cute::X>{});
        constexpr auto bM = cute::get<0>(typename BlockGEMM::BlockTiler{});
        constexpr auto threads = BlockGEMM::GEMM::block_dim.x;
        constexpr auto elems = SHARED_SIZE / (threads * sizeof(ElementSource));
        static_assert(rScratch.size() == elems);
        static_assert(bN % elems == 0);
        constexpr auto trips = bN / elems;

        // Transposed layout
        constexpr auto sCLay = cute::make_layout(cute::Shape<cute::Int<bM>, cute::Int<elems>>{}, cute::LayoutRight{});
        const auto sC = cute::make_tensor(cute::make_smem_ptr(CAST_TO(ElementSource, workspace)), sCLay);

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
                registers(j + i * elems) = sC(threadIdx.x, j);
            }
        }

        if (threadIdx.x < tileSize) {
            #pragma unroll
            for (uint i = 0; i < bN; ++i) {
                atomicAdd(&activations(tokenIdx, i), registers(i));
            }
        }
    }
};

template<unsigned int M, unsigned int K, typename Element>
__global__ __maxnreg__(128) void deviceCombine(cuda::std::byte* __restrict__ p) {
    using BlockGEMM = BlockMM<Element, Element, float, 800>;
    constexpr auto tiler = typename BlockGEMM::BlockTiler{};
    constexpr auto bN = cute::get<1>(tiler);
    constexpr auto threads = BlockGEMM::GEMM::block_dim.x;
    constexpr auto elems = SHARED_SIZE / (threads * sizeof(Element));
    const auto* __restrict__ tokenIndices = p;
    const auto* __restrict__ inputs = p + sizeof(unsigned int) * M;
    const auto activations = make_tensor(cute::make_gmem_ptr(p + sizeof(unsigned int) * M + sizeof(Element) * M * K),
        make_layout(cute::make_shape(M, K), cute::LayoutRight{}));
    // constexpr Combine<Modality::coalesce, 800> combineOp{};
    extern __shared__ Element scratch[];
    cutlass::AlignedArray<Element, bN> regs{};
    cutlass::AlignedArray<Element, elems> rS{};
    // combineOp(scratch, tokenIndices, regs, rS, inputs, activations, M, K, blockIdx.x, M);
}
__host__ __forceinline__
void hostCombine() {
    const auto playStream = cudaStreamPerThread;
    constexpr auto M = 128;
    constexpr auto K = 64;
    using inputValueType = cute::half_t;

    constexpr auto aSize = sizeof(inputValueType) * M * K;
    constexpr auto len = M * sizeof(unsigned int) + 2 * aSize;
    cuda::std::byte* a;
    CHECK_ERROR_EXIT(cudaMallocAsync(&a, len, playStream));
    auto* data = static_cast<cuda::std::byte*>(calloc(len, sizeof(cuda::std::byte)));
    for (uint i = 0; i < M; ++i) {
        CAST_TO(unsigned int, data)[i] = i;
    }
    data += sizeof(unsigned int) * M;

    for (uint i = 0; i < M; ++i) {
        for (uint j = 0; j < K; ++j) {
            CAST_TO(inputValueType, data)[j + i * M] = inputValueType{i};
        }
    }
    data += sizeof(inputValueType) * M * K;
    for (uint i = 0; i < M; ++i) {
        for (uint j = 0; j < K; ++j) {
            CAST_TO(inputValueType, data)[j + i * M] = inputValueType{i};
        }
    }
    CHECK_ERROR_EXIT(cudaMemcpyAsync(a, data, aSize, cudaMemcpyHostToDevice, playStream));
    deviceCombine<M, K, inputValueType><<<1, 128, SHARED_SIZE, playStream>>>(a);
    CHECK_ERROR_EXIT(cudaFreeAsync(a, playStream));
    free(data);
    CHECK_LAST();
}
#endif //COMBINE_CUH
