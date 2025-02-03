#include <array>
#include <iostream>

#include <cublasdx.hpp>
#include <cutlass/numeric_conversion.h>
#include <torch/torch.h>

#include "util.cuh"

struct Expert final : torch::nn::Module {
    torch::nn::Linear g1;
    torch::nn::Linear g2;

    Expert():
          g1(2, 4),
          g2(4, 2) {
        register_module("g1", g1);
        register_module("g2", g2);
    }

    torch::Tensor forward(torch::Tensor& x) {
        x = relu(g1->forward(x));
        return g2->forward(x);
    }
};

__global__ void theatre() {
    const auto x = 0.5_tf32;
    const auto y = 0.25_tf32;
    printf("Result is %f", __fdividef(x, y));
}

__host__ __forceinline__
void tensorWork() {
    std::array<float, 4> a{0, 1, 2, 3};
    torch::Device device(torch::kCUDA);
    const torch::Tensor tensor = torch::from_blob(a.data(), {2,2}).to(device).to(torch::kFloat8_e4m3fn);
    switch (tensor.scalar_type()) {
        case torch::kFloat: {
            // tf32 is automatic
            if (at::globalContext().allowTF32CuBLAS() || at::globalContext().allowTF32CuDNN()) {
                std::cout << "This is tf32!" << std::endl;
            }
            else {
                std::cout << "This is float!" << std::endl;
            }
        }
        break;
        case torch::kBFloat16:
            std::cout << "This is bf16!" << std::endl;
        break;
        case torch::kFloat16:
            std::cout << "This is fp16!" << std::endl;
        break;
        case torch::kFloat8_e4m3fn:
            std::cout << "This is fp8e4!" << std::endl;
        break;
        case torch::kFloat8_e5m2:
            std::cout << "This is fp8e5!" << std::endl;
        break;
        default:
            std::cout << "This is weird!" << std::endl;
    }
    if (tensor.dtype() == torch::kFloat) {
        std::cout << "This is float!" << std::endl;
    }
    else {
        std::cout << "This is not float!" << std::endl;
    }
    const Expert model;
    const Expert model2;
    std::cout << model << std::endl;
    for (const auto& p : model.named_parameters()) {
        std::cout << p.key() << std::endl;
        std::cout << p.value() << std::endl;
    }
    // pack both experts into a single torch tensor
    constexpr auto nX = 2U;
    constexpr auto GEMMs = 2U;
    constexpr auto h = 2U;
    constexpr auto upH = 4U;
    const torch::Tensor pT = torch::zeros({nX, GEMMs, h, upH});
    pT[0][0] = model.named_parameters()[2].value();
    const auto expert1 = pT[0][0];
    std::cout << expert1 << std::endl;
    std::cout << pT.is_contiguous() << std::endl;
}

int main() {
    tensorWork();
}
