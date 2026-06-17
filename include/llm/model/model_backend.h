#pragma once

#include "llm/core/token.h"

#include <vector>

namespace llm {

class ModelBackend {
public:
    virtual ~ModelBackend() = default;

    virtual size_t vocabSize() const = 0;

    virtual std::vector<float> forwardNextToken(
        const std::vector<TokenId>& contextTokens
    ) = 0;
};

} // namespace llm
