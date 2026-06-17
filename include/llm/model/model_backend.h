#pragma once

#include "llm/core/token.h"

#include <stdexcept>
#include <vector>

namespace llm {

class ModelBackend {
public:
    virtual ~ModelBackend() = default;

    virtual size_t vocabSize() const = 0;

    virtual std::vector<float> forwardNextToken(
        const std::vector<TokenId>& contextTokens
    ) = 0;

    // For now this is very naive, simple loops
    virtual std::vector<std::vector<float>> forwardNextTokenBatch(
        const std::vector<std::vector<TokenId>>& batchContexts
    ) {
        std::vector<std::vector<float>> batchLogits;
        batchLogits.reserve(batchContexts.size());

        for (const auto& context : batchContexts) {
            if (context.empty()) {
                throw std::runtime_error("ModelBackend received empty context");
            }

            batchLogits.push_back(forwardNextToken(context));
        }

        return batchLogits;
    }
};


} // namespace llm
