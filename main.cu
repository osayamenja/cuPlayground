#include <algorithm>
#include <random>

#include <fmt/ranges.h>
#include <fmt/core.h>

#include "mma.cuh"
#include "util.cuh"

// Source for floatEqual: https://stackoverflow.com/a/253874
template<typename F> requires cuda::std::is_floating_point_v<F>
__forceinline__
bool floatEqual(const F& a, const F& b){
    return fabs(a - b) <=
           ( (fabs(a) > fabs(b) ? fabs(b) : fabs(a)) * cuda::std::numeric_limits<F>::epsilon());
}

struct Edge{
    unsigned int node1;
    unsigned int node2;
    float weight;

    Edge() = default;
    /// Order is important!
    Edge(const unsigned int& _node1, const unsigned int& _node2, const float& _weight):
    node1(_node1), node2(_node2), weight(_weight){}

    __forceinline__
    unsigned int getNeighbor(const unsigned int& other) const {
        assert(node1 == other || node2 == other);
        return (node1 == other)? node2 : node1;
    }

    __forceinline__
    bool operator==(const Edge& other) const {
        return this->node1 == other.node1 && this->node2 == other.node2;
    }

    __forceinline__
    bool operator!=(const Edge& other) const {
        return !(*this == other);
    }

    __forceinline__
    bool operator<(const Edge& other) const {
        if(floatEqual(this->weight, other.weight)){
            if(this->node1 == other.node1){
                return this->node2 < other.node2;
            }
            else{
                return this->node1 < other.node1;
            }
        }
        return this->weight < other.weight;
    }

    __forceinline__
    bool operator<=(const Edge& other) const {
        return *this < other || *this == other;
    }

    __forceinline__
    bool operator>(const Edge& other) const {
        return !(*this <= other);
    }

    __forceinline__
    bool operator>=(const Edge& other) const {
        return *this > other || *this == other;
    }

    __forceinline__
    std::string toString() const {
        return "{"
               "\"weight\": " + std::to_string(weight)
               + ", \"node1\": " + std::to_string(node1)
               + ", \"node2\": " + std::to_string(node2) + "}";
    }

    __forceinline__
    bool isLimboEdge() const{
        /// Define a self-edge with zero weight as limbo or null edge
        return node1 == node2 && floatEqual(weight, 0.0f);
    }

    __forceinline__
    static Edge limboEdge() {
        return {0,0,0.0f};
    }
};

int main() {
    std::random_device rd;
    std::mt19937 g(rd());

    constexpr auto dim = 4;
    auto print = [](const Edge& e) { std::cout << e.toString() << std::endl; };
    std::vector<Edge> edges(dim * (dim - 1));
    uint k = 0U;
    for (uint i = 0; i < dim; ++i) {
        for (uint j = 0; j < dim; ++j) {
            if (i != j)[[likely]] {
                edges[k++] = {i, j, static_cast<float>(i + j)};
            }
        }
    }
    printf("--------------------After insertion---------------\n");
    std::ranges::for_each(edges.begin(), edges.end(), print);
    std::ranges::shuffle(edges.begin(), edges.end(), g);
    printf("--------------------Shuffled---------------\n");
    std::ranges::for_each(edges.begin(), edges.end(), print);
    std::ranges::sort(edges.begin(), edges.end(), std::less{});
    printf("--------------------Sorted---------------\n");
    std::ranges::for_each(edges.begin(), edges.end(), print);
    //testCollective()
}
