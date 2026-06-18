#pragma once

#include "llm/core/token.h"

#include <vector>

namespace llm {

class ModelBackend {
public:
    virtual ~ModelBackend() = default;

    virtual size_t vocabSize() const = 0;

    // prefillBatch(...) -> logits for first generated token
    virtual std::vector<std::vector<float>> prefillBatch(
        const std::vector<std::vector<TokenId>>& batchPromptTokens
    ) = 0;

    // decodeBatch(...)  -> logits for next generated token
    virtual std::vector<std::vector<float>> decodeBatch(
        const std::vector<TokenId>& lastTokens,
        const std::vector<int>& positions
    ) = 0;
};


} // namespace llm
