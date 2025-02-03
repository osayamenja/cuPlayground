#include <array>
#include <iostream>

#include <torch/torch.h>

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

int main() {
    std::array<float, 4> a{0, 1, 2, 3};
    torch::Device device(torch::kCUDA);
    const torch::Tensor tensor = torch::from_blob(a.data(), {2,2}).to(device);
    const Expert model;
    std::cout << model << std::endl;
    for (const auto& p : model.named_parameters()) {
        std::cout << p.key() << std::endl;
        std::cout << p.value() << std::endl;
        std::cout << p.value().data()[0] << std::endl;
    }
    std::cout << tensor << std::endl;
}
