#pragma once

#include "llm/core/token.h"
#include "llm/core/sequence.h"

#include <stdexcept>
#include <vector>

namespace llm {

/*
PrefillBatch:
  full prompt tokens

DecodeBatch:
  only last token + current position
*/

struct PrefillBatch {
    std::vector<RequstId> requestIds;
    std::vector<std::vector<TokenId>> promptTokens;

    size_t size() const {
        return requestIds.size();
    }

    bool empty() const {
        return requestIds.empty();
    }

    void validate() const {
        if (requestIds.size() != promptTokens.size()) {
            throw std::runtime_error("PrefillBatch size mismatch");
        }

        for (const auto& prompt : promptTokens) {
            if (prompt.empty()) {
                throw std::runtime_error("PrefillBatch contains empty prompt");
            }
        }
    }
};

struct DecodeBatch {
    std::vector<RequstId> requestIds;

    // One input token per active sequence
    std::vector<TokenId> lastTokens;

    // Current total sequence length before predicting the next token
    std::vector<int> positions;

    size_t size() const {
        return requestIds.size();
    }

    bool empty() const {
        return requestIds.empty();
    }

    void validate() const {
        if (requestIds.size() != lastTokens.size() ||
            requestIds.size() != positions.size()) {
            throw std::runtime_error("DecodeBatch size mismatch");
        }
    }
};


} // namespace llm
