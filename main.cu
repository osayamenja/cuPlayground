#include <cuda/std/cassert>
#include <cute/tensor.hpp>

#include "util.cuh"
#include "auditorium/combine.cuh"

__host__ __forceinline__
void hostTiling(){
    using tiler = cute::Shape<cute::Int<4>, cute::Int<2>>;
    constexpr auto M = 8;
    constexpr auto K = 8;
    const auto mA = make_counting_tensor(cute::Layout<cute::Shape<cute::Int<M>, cute::Int<K>>>{});
    constexpr auto tilesM = cute::get<0>(mA.shape()) / 4;
    constexpr auto tilesK = cute::get<1>(mA.shape()) / 2;
    constexpr auto tileCoord = idx2crd(1, cute::Shape(tilesM, tilesK), cute::Stride(tilesK ,1));
    constexpr auto ctaCoord = cute::make_coord(cute::get<0>(tileCoord), cute::get<1>(tileCoord));
    const auto gA = local_tile(mA, tiler{}, ctaCoord);
    print_tensor(mA); printf("\n");
    print_tensor(gA);
}

int main() {
    /*using x = cuda::std::integral_constant<unsigned int, 4>;
    testCollective();*/
    hostTiling();
    //runScheduler<SchedulerType::vanilla>();
    //runScheduler<SchedulerType::fast>();
}