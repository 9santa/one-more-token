#include "llm/cuda/vector_add.h"

#include <iostream>
#include <vector>

int main() {
    std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> b = {10.0f, 20.0f, 30.0f, 40.0f};

    std::vector<float> out = llm::cudaVectorAdd(a, b);

    for (float x : out) {
        std::cout << x << " ";
    }
    std::cout << "\n";

    return 0;
}
