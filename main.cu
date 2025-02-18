#include <array>
#include <iostream>

#include <cuda/std/array>
#include <cuda/std/tuple>
#include <cuda/barrier>
#include <cublasdx.hpp>
#include <torch/torch.h>

#include "util.cuh"
//#include "mma.cuh"
struct Expert final : torch::nn::Module {
    torch::nn::Linear g1;
    torch::nn::Linear g2;

    Expert():
          g1(torch::nn::LinearOptions(2, 4)),
          g2(torch::nn::LinearOptions(4, 2)) {
        register_module("g1", g1);
        register_module("g2", g2);
    }

    torch::Tensor forward(torch::Tensor& x) {
        x = relu(g1->forward(x));
        return g2->forward(x);
    }
};

struct __align__(8) Foo {
    uint x;
    uint y;
};

__host__ __forceinline__
void tensorWork() {
    const torch::nn::Sequential expert(
        torch::nn::Linear(2,4),
        torch::nn::ReLU(),
        torch::nn::Linear(4, 2)
        );
    std::cout << expert << std::endl;

    std::array<float, 4> a{0, 1, 2, 3};
    torch::Device device(torch::kCUDA);
    const torch::Tensor tensor = torch::from_blob(a.data(), {2,2}).to(device).to(torch::kFloat8_e4m3fn);
    const Expert model;
    const Expert model2;
    std::cout << model << std::endl;
    for (const auto& p : model.named_parameters()) {
        // [0.weight, 0.bias, ...]
        std::cout << p.key() << std::endl;
        std::cout << p.value() << std::endl;
    }
    // pack both experts into a single torch tensor
    constexpr auto nX = 2U;
    constexpr auto GEMMs = 2U;
    constexpr auto h = 2U;
    constexpr auto upH = 4U;
    const torch::Tensor pT = torch::zeros({nX, GEMMs, h, upH}).contiguous();
    pT[0][0] = model.named_parameters()[2].value();
    pT[0][1] = model.named_parameters()[2].value();
    const auto expert1 = pT[0][0];

    // flatten [s, b, h] -> [sb, h]
    constexpr auto s = 2U;
    constexpr auto b = 4U;
    const torch::Tensor act = torch::ones({s, b, h}).contiguous();
    const auto sz = act.sizes();
    /*std::cout << sz.size() << std::endl;
    std::cout << act << std::endl;
    std::cout << act.view({sz[0]*sz[1], h}) << std::endl;*/

    /*auto* __restrict__ p = pT.const_data_ptr<float>();
    std::cout << p[15] << std::endl;*/

}
int main() {
    using mma = cute::TiledMMA<
          cute::MMA_Atom<cute::SM80_16x8x8_F32TF32TF32F32_TN>,
          cute::Layout<cute::Shape<cute::_2, cute::_2, cute::_1>>
    >;
    print_latex(mma{});

}
