#pragma once

#include "llm/core/token.h"
#include "llm/core/generation_config.h"

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace llm {

class Sampler {
public:
    TokenId sampleGreedy(const std::vector<float>& logits) const {
        if (logits.empty()) {
            throw std::runtime_error("Cannot sample from empty logits vector");
        }

        auto bestIt = std::max_element(logits.begin(), logits.end());
        auto bestId = std::distance(logits.begin(), bestIt);
        float bestValue = *bestIt;

        return bestId;
    }

    TokenId sample(const std::vector<float>& logits,
                   const GenerationConfig& config) const {
        (void)config; // TODO later

        return sampleGreedy(logits);
    }
};


} // namespace llm
