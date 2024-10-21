//
// Created by oja7 on 10/10/24.
//

#ifndef GEMM_CUH
#define GEMM_CUH

#include <cublasdx.hpp>

template<class GEMM>
__global__ void gemm_kernel(const typename GEMM::c_value_type  alpha,
                            const typename GEMM::a_value_type* a,
                            const typename GEMM::b_value_type* b,
                            const typename GEMM::c_value_type  beta,
                            typename GEMM::c_value_type* c, bool skip = false) {
    extern __shared__ __align__(16) char smem[];

    // Make global memory tensor
    auto a_global_tensor = cublasdx::make_tensor(a, GEMM::get_layout_gmem_a());
    auto b_global_tensor = cublasdx::make_tensor(b, GEMM::get_layout_gmem_b());
    auto c_global_tensor = cublasdx::make_tensor(c, GEMM::get_layout_gmem_c());
    if (cute::thread0() && !skip) {
        cute::print_tensor(a_global_tensor);
        cute::print_tensor(b_global_tensor);
    }

    // Make shared memory tensor
    auto [smem_a, smem_b, smem_c] = GEMM::slice_shared_memory(smem);
    auto a_shared_tensor = cublasdx::make_tensor(smem_a, GEMM::suggest_layout_smem_a());
    auto b_shared_tensor = cublasdx::make_tensor(smem_b, GEMM::suggest_layout_smem_b());
    auto c_shared_tensor = cublasdx::make_tensor(smem_c, GEMM::suggest_layout_smem_c());

    // Load data from global memory tensor to shared memory tensor
    using alignment = cublasdx::alignment_of<GEMM>;
    cublasdx::copy<GEMM, alignment::a>(a_global_tensor, a_shared_tensor);
    cublasdx::copy<GEMM, alignment::b>(b_global_tensor, b_shared_tensor);
    cublasdx::copy<GEMM, alignment::c>(c_global_tensor, c_shared_tensor);
    cublasdx::copy_wait();

    // Execute GEMM
    GEMM().execute(alpha, a_shared_tensor, b_shared_tensor, beta, c_shared_tensor);
    if (cute::thread0() && !skip) {
        cute::print_tensor(c_shared_tensor);
    }

    __syncthreads();
    // Store data from shared memory tensor to global memory tensor
    cublasdx::copy<GEMM, alignment::c>(c_shared_tensor, c_global_tensor);
}

template<unsigned int Arch=800>
int introduction_example() {
    constexpr auto M = 2U;
    constexpr auto N = 2U;
    constexpr auto K = 2U;
    using value_type = float;
    using GEMM = decltype(cublasdx::Size<M, N, K>()
                  + cublasdx::Precision<value_type>()
                  + cublasdx::Type<cublasdx::type::real>()
                  + cublasdx::Arrangement<cublasdx::row_major>()
                  + cublasdx::Function<cublasdx::function::MM>()
                  + cublasdx::SM<Arch>()
                  + cublasdx::Block());
    static_assert(cublasdx::is_complete_blas_execution<GEMM>::value);

    constexpr auto global_a_size = M*K;
    constexpr auto global_b_size = K*N;
    constexpr auto global_c_size = M*N;

    // Allocate managed memory for A, B, C matrices in one go
    value_type* abc;
    constexpr auto        size       = global_a_size + global_b_size + global_c_size;
    constexpr auto        size_bytes = size * sizeof(value_type);
    CUTE_CHECK_ERROR(cudaMallocManaged(&abc, size_bytes));
    // Generate data
    for (uint i = 0; i < global_a_size + global_b_size; i++) {
        abc[i] = static_cast<value_type>(i);
    }

    const value_type* a = abc;
    const value_type* b = abc + global_a_size;
    value_type* c = abc + global_a_size + global_b_size;

    // Invokes kernel with GEMM::block_dim threads in CUDA block
    gemm_kernel<GEMM><<<1, GEMM::block_dim, GEMM::shared_memory_size>>>(1.0f, a, b, 0.0f, c);
    CUTE_CHECK_ERROR(cudaPeekAtLastError());
    CUTE_CHECK_ERROR(cudaDeviceSynchronize());
    std::array<value_type, M*N> result {};
    CUTE_CHECK_ERROR(cudaMemcpy(result.data(), c, sizeof(value_type)*global_c_size, cudaMemcpyDeviceToHost));
    CUTE_CHECK_ERROR(cudaFree(abc));
    std::cout << "Success" << std::endl;
    fmt::println("GEMM Results {}", result);
    return 0;
}
#endif //GEMM_CUH
