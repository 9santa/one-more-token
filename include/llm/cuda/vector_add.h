#pragma once

#include <vector>

namespace llm {

std::vector<float> cudaVectorAdd(const std::vector<float>& a,
                                 const std::vector<float>& b);

} // namespace llm
